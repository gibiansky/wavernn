
#include "gemv.h"

#include <immintrin.h>
#include <mkl.h>
#include <omp.h>
#include <torch/script.h>

#include "wavernn_assert.h"

constexpr int BLOCK_SIZE = 8;

using torch::Tensor;

namespace wavernn::fallback {

void gemv(Tensor& out, const Tensor& bias, const Tensor& matrix,
          const Tensor& vector) {
  at::addmv_out(out, bias, matrix, vector);
}

}  // namespace wavernn::fallback

#ifdef __AVX2__
namespace wavernn::avx2 {

/// Sum up the values in an AVX 256-bit XMM register and return them as a float.
/// Implementation courageously borrowed from Marat Dukhan's StackOverflow
/// answer:
///
///     https://stackoverflow.com/a/13222410
///
/// @param x XMM register with contents to sum horizontally.
/// @returns The sum of the values in the register.
inline float mm256_horizontal_sum(__m256 x) {
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

/**
 * Compute a sparse matrix-vector multiply.
 *
 * @param output_size The number of output rows in this gemv.
 * @param output Where to write the output (dense float32 vector).
 * @param input Where to read the input data from (dense float32 vector).
 * @param weights Packed weights.
 * @param bias The bias vector (dense float32 vector).
 * @param blocksPerRow Number of blocks involved in each row multiply.
 * @param rowOffsets The offset in the weights for the nth row.
 * @param indices The indices in the input vector to load blocks from.
 */
void sparse_gemv(int output_size, float* const __restrict__ output,
                 const float* const __restrict__ input,
                 const float* const __restrict__ weights,
                 const float* const __restrict__ bias,
                 const int* const __restrict__ blocksPerRow,
                 const int* const __restrict__ rowOffsets,
                 const int* const __restrict__ indices) {
  // Parallelize the computation across the output row. Each thread can
  // separately compute outputs for a block of the output vector.
#pragma omp parallel for
  for (int output_idx = 0; output_idx < output_size; output_idx++) {
    // Number of block multiplications for this row.
    int n_blocks = blocksPerRow[output_idx];

    // Where the weights for this row start.
    const float* block_weights = weights + rowOffsets[output_idx];

    // Where the input indices for this row start.
    const int* block_indices = indices + rowOffsets[output_idx] / BLOCK_SIZE;

    // Unroll the block multiplication loop by 4. This speeds up the gemvs by a
    // little bit. If the number of blocks is not a multiple of 4, the follow-up
    // loop will take care of it.
    int block_idx = 0;
    int n_blocks_grouped = n_blocks - (n_blocks % 4);
    __m256 sum_vec_0 = _mm256_set1_ps(0);
    __m256 sum_vec_1 = sum_vec_0;
    __m256 sum_vec_2 = sum_vec_0;
    __m256 sum_vec_3 = sum_vec_0;
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      // Load the index of each block of the input.
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      // Load the data for the 4 input blocks.
      __m256 input_vec_0 = _mm256_loadu_ps(input + input_idx_0);
      __m256 input_vec_1 = _mm256_loadu_ps(input + input_idx_1);
      __m256 input_vec_2 = _mm256_loadu_ps(input + input_idx_2);
      __m256 input_vec_3 = _mm256_loadu_ps(input + input_idx_3);

      // Load the data for the 4 weight blocks.
      __m256 weight_vec_0 =
          _mm256_loadu_ps(block_weights + (block_idx + 0) * BLOCK_SIZE);
      __m256 weight_vec_1 =
          _mm256_loadu_ps(block_weights + (block_idx + 1) * BLOCK_SIZE);
      __m256 weight_vec_2 =
          _mm256_loadu_ps(block_weights + (block_idx + 2) * BLOCK_SIZE);
      __m256 weight_vec_3 =
          _mm256_loadu_ps(block_weights + (block_idx + 3) * BLOCK_SIZE);

      // Perform multiplications and accumulate into the accumulators.
      sum_vec_0 = _mm256_fmadd_ps(input_vec_0, weight_vec_0, sum_vec_0);
      sum_vec_1 = _mm256_fmadd_ps(input_vec_1, weight_vec_1, sum_vec_1);
      sum_vec_2 = _mm256_fmadd_ps(input_vec_2, weight_vec_2, sum_vec_2);
      sum_vec_3 = _mm256_fmadd_ps(input_vec_3, weight_vec_3, sum_vec_3);
    }

    // Sum up accumulators to get an accumulator for the follow-up loop.
    __m256 sum_vec = _mm256_add_ps(_mm256_add_ps(sum_vec_0, sum_vec_1),
                                   _mm256_add_ps(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      // Load the input block.
      __m256 input_vec = _mm256_loadu_ps(input + block_indices[block_idx]);

      // Load the weight block.
      __m256 weight_vec =
          _mm256_loadu_ps(block_weights + block_idx * BLOCK_SIZE);

      // Multiply and accumulate.
      sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
    }

    // Compute the horizontal sum, add in the bias, and write to the output.
    output[output_idx] = bias[output_idx] + mm256_horizontal_sum(sum_vec);
  }
}

}  // namespace wavernn::avx2
#endif

namespace wavernn {

PackedLinear::PackedLinear(const Tensor& matrix, const Tensor& bias)
    : matrix_(matrix), bias_(bias) {
  output_size_ = matrix.size(0);
  input_size_ = matrix.size(1);

  ASSERT_BOOL(output_size_ % BLOCK_SIZE == 0);
  ASSERT_BOOL(input_size_ % BLOCK_SIZE == 0);
  ASSERT_TENSOR_SIZE(matrix, output_size_, input_size_);
  ASSERT_TENSOR_SIZE(bias, output_size_);

  // Repack the block-sparse matrix. To repack the matrix, we go through the
  // matrix in blocks and check if the block is nonzero. If the block is
  // nonzero, then add its weights to the linearly growing weight vector, and
  // record the location in the input where we would multiply this by. Record
  // the number of blocks needed for each row.
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

void PackedLinear::gemv(Tensor& out, const Tensor& vector) const {
  ASSERT_TENSOR_SIZE(out, output_size_);
  ASSERT_TENSOR_SIZE(vector, input_size_);
#ifdef __AVX2__
  float* const __restrict__ output = out.data_ptr<float>();
  const float* const __restrict__ input = vector.data_ptr<float>();
  const float* const __restrict__ weights = data_.data();
  const float* const __restrict__ bias = bias_.data_ptr<float>();
  const int* const __restrict__ blocksPerRow = blocksPerRow_.data();
  const int* const __restrict__ rowOffsets = rowOffsets_.data();
  const int* const __restrict__ indices = indices_.data();
  wavernn::avx2::sparse_gemv(output_size_, output, input, weights, bias,
                             blocksPerRow, rowOffsets, indices);
#else
  wavernn::fallback::gemv(out, bias_, matrix_, vector);
#endif
}

}  // namespace wavernn
