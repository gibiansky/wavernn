
#include <immintrin.h>
#include <mkl/mkl.h>
#include <omp.h>
#include <torch/script.h>

#include "kernel.h"

constexpr int BLOCK_SIZE = 8;

using torch::Tensor;

/// Sum up the values in an AVX 256-bit XMM register and return them as a float.
/// Implementation courageously borrowed from Marat Dukhan's StackOverflow
/// answer:
///
///     https://stackoverflow.com/a/13222410
///
/// @param x XMM register with contents to sum horizontally.
/// @returns The sum of the values in the register.
inline float sum8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const auto hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const auto loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const auto sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const auto loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const auto hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const auto sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const auto lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const auto hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const auto sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

SparsePackedMatrix::SparsePackedMatrix(const Tensor& matrix, const Tensor& bias)
    : matrix_(matrix), bias_(bias) {
  output_size_ = matrix.size(0);
  input_size_ = matrix.size(1);

  ASSERT_BOOL(output_size_ % BLOCK_SIZE == 0);
  ASSERT_BOOL(input_size_ % BLOCK_SIZE == 0);
  ASSERT_TENSOR_SIZE(matrix, output_size_, input_size_);
  ASSERT_TENSOR_SIZE(bias, output_size_);

  auto mat = matrix.accessor<float, 2>();
  for (int r = 0; r < output_size_; r++) {
    rowOffsets_.push_back(data_.size());

    int numBlocks = 0;
    for (int c = 0; c < input_size_; c += BLOCK_SIZE) {
      bool empty = true;
      for (int i = c; i < c + BLOCK_SIZE; i++) {
        if (mat[r][i] != 0) {
          empty = false;
          break;
        }
      }

      if (!empty) {
        numBlocks++;
        indices_.push_back(c);
        for (int i = c; i < c + BLOCK_SIZE; i++) {
          data_.push_back(mat[r][i]);
        }
      }
    }

    blocksPerRow_.push_back(numBlocks);
  }
}

void SparsePackedMatrix::gemv(const Tensor& out, const Tensor& vector) const {
  ASSERT_TENSOR_SIZE(out, output_size_);
  ASSERT_TENSOR_SIZE(vector, input_size_);

  float* const __restrict__ output = out.data_ptr<float>();
  const float* const __restrict__ input = vector.data_ptr<float>();
  const float* const __restrict__ weights = data_.data();
  const float* const __restrict__ bias = bias_.data_ptr<float>();
  const int* const __restrict__ blocksPerRow = blocksPerRow_.data();
  const int* const __restrict__ rowOffsets = rowOffsets_.data();
  const int* const __restrict__ indices = indices_.data();

#pragma omp parallel for
  for (int output_idx = 0; output_idx < output_size_; output_idx++) {
    int n_blocks = blocksPerRow[output_idx];
    int weight_offset = rowOffsets[output_idx];
    int input_offset = weight_offset / BLOCK_SIZE;
    const int* __restrict__ block_indices = indices + input_offset;

    int block_idx = 0;
    int n_blocks_grouped = n_blocks - (n_blocks % 4);
    __m256 sum_vec_0 = _mm256_set1_ps(0);
    __m256 sum_vec_1 = sum_vec_0;
    __m256 sum_vec_2 = sum_vec_0;
    __m256 sum_vec_3 = sum_vec_0;
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      __m256 input_vec_0 = _mm256_loadu_ps(input + input_idx_0);
      __m256 input_vec_1 = _mm256_loadu_ps(input + input_idx_1);
      __m256 input_vec_2 = _mm256_loadu_ps(input + input_idx_2);
      __m256 input_vec_3 = _mm256_loadu_ps(input + input_idx_3);

      __m256 weight_vec_0 = _mm256_loadu_ps(weights + weight_offset +
                                            (block_idx + 0) * BLOCK_SIZE);
      __m256 weight_vec_1 = _mm256_loadu_ps(weights + weight_offset +
                                            (block_idx + 1) * BLOCK_SIZE);
      __m256 weight_vec_2 = _mm256_loadu_ps(weights + weight_offset +
                                            (block_idx + 2) * BLOCK_SIZE);
      __m256 weight_vec_3 = _mm256_loadu_ps(weights + weight_offset +
                                            (block_idx + 3) * BLOCK_SIZE);

      sum_vec_0 = _mm256_fmadd_ps(input_vec_0, weight_vec_0, sum_vec_0);
      sum_vec_1 = _mm256_fmadd_ps(input_vec_1, weight_vec_1, sum_vec_1);
      sum_vec_2 = _mm256_fmadd_ps(input_vec_2, weight_vec_2, sum_vec_2);
      sum_vec_3 = _mm256_fmadd_ps(input_vec_3, weight_vec_3, sum_vec_3);
    }

    __m256 sum_vec = _mm256_add_ps(_mm256_add_ps(sum_vec_0, sum_vec_1),
                                   _mm256_add_ps(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      int input_idx = block_indices[block_idx];
      __m256 input_vec = _mm256_loadu_ps(input + input_idx);
      __m256 weight_vec =
          _mm256_loadu_ps(weights + weight_offset + block_idx * BLOCK_SIZE);
      sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
    }
    output[output_idx] = bias[output_idx] + sum8(sum_vec);
  }
}
