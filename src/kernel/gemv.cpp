
#include "gemv.h"

#if defined(__AVX2__)
#include <immintrin.h>
#include <mkl.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <omp.h>
#include <torch/script.h>

#include "wavernn_assert.h"

#if defined(__AVX512F__)
constexpr int BLOCK_SIZE = 16;
constexpr int QUANTIZED_BLOCK_SIZE = 32;
constexpr float QUANTIZATION_SCALE = 16384.0f;
#else
constexpr int BLOCK_SIZE = 8;
constexpr int QUANTIZED_BLOCK_SIZE = 16;
constexpr float QUANTIZATION_SCALE = 16384.0f;
#endif

using torch::Tensor;

namespace wavernn::fallback {

void gemv(Tensor& out, const Tensor& bias, const Tensor& matrix,
          const Tensor& vector) {
  at::addmv_out(out, bias, matrix, vector);
}

}  // namespace wavernn::fallback
#if defined(__ARM_NEON)
namespace wavernn::neon {
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

    const float* input_1 = input + 4;
    const float* block_weights_1 = block_weights + 4;

    // Unroll the block multiplication loop by 4. This speeds up the gemvs by a
    // little bit. If the number of blocks is not a multiple of 4, the follow-up
    // loop will take care of it.
    int block_idx = 0;
    int n_blocks_grouped = n_blocks - (n_blocks % 4);
    float32x4_t sum_vec_0 = vdupq_n_f32(0.0);
    float32x4_t sum_vec_1 = vdupq_n_f32(0.0);
    float32x4_t sum_vec_2 = vdupq_n_f32(0.0);
    float32x4_t sum_vec_3 = vdupq_n_f32(0.0);
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      // Load the index of each block of the input.
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      // Load the data for the 4 input blocks.
      float32x4_t input_vec_0_0 = vld1q_f32(input + input_idx_0);
      float32x4_t input_vec_0_1 = vld1q_f32(input_1 + input_idx_0);
      float32x4_t input_vec_1_0 = vld1q_f32(input + input_idx_1);
      float32x4_t input_vec_1_1 = vld1q_f32(input_1 + input_idx_1);
      float32x4_t input_vec_2_0 = vld1q_f32(input + input_idx_2);
      float32x4_t input_vec_2_1 = vld1q_f32(input_1 + input_idx_2);
      float32x4_t input_vec_3_0 = vld1q_f32(input + input_idx_3);
      float32x4_t input_vec_3_1 = vld1q_f32(input_1 + input_idx_3);

      // Load the data for the 4 weight blocks.
      float32x4_t weight_vec_0_0 = vld1q_f32(block_weights + (block_idx + 0) * BLOCK_SIZE);
      float32x4_t weight_vec_0_1 = vld1q_f32(block_weights_1 + (block_idx + 0) * BLOCK_SIZE);
      float32x4_t weight_vec_1_0 = vld1q_f32(block_weights + (block_idx + 1) * BLOCK_SIZE);
      float32x4_t weight_vec_1_1 = vld1q_f32(block_weights_1 + (block_idx + 1) * BLOCK_SIZE);
      float32x4_t weight_vec_2_0 = vld1q_f32(block_weights + (block_idx + 2) * BLOCK_SIZE);
      float32x4_t weight_vec_2_1 = vld1q_f32(block_weights_1 + (block_idx + 2) * BLOCK_SIZE);
      float32x4_t weight_vec_3_0 = vld1q_f32(block_weights + (block_idx + 3) * BLOCK_SIZE);
      float32x4_t weight_vec_3_1 = vld1q_f32(block_weights_1 + (block_idx + 3) * BLOCK_SIZE);

      // Perform multiplications and accumulate into the accumulators.
      sum_vec_0 = vmlaq_f32(vmlaq_f32(sum_vec_0, weight_vec_0_0, input_vec_0_0), weight_vec_0_1, input_vec_0_1);
      sum_vec_1 = vmlaq_f32(vmlaq_f32(sum_vec_1, weight_vec_1_0, input_vec_1_0), weight_vec_1_1, input_vec_1_1);
      sum_vec_2 = vmlaq_f32(vmlaq_f32(sum_vec_2, weight_vec_2_0, input_vec_2_0), weight_vec_2_1, input_vec_2_1);
      sum_vec_3 = vmlaq_f32(vmlaq_f32(sum_vec_3, weight_vec_3_0, input_vec_3_0), weight_vec_3_1, input_vec_3_1);
    }

    // Sum up accumulators to get an accumulator for the follow-up loop.
    float32x4_t sum_vec = vaddq_f32(vaddq_f32(sum_vec_0, sum_vec_1), vaddq_f32(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      // Load the input block.
      float32x4_t input_vec_0 = vld1q_f32(input + block_indices[block_idx]);
      float32x4_t input_vec_1 = vld1q_f32(input_1 + block_indices[block_idx]);

      // Load the weight block.
      float32x4_t weight_vec_0 = vld1q_f32(block_weights + block_idx * BLOCK_SIZE);
      float32x4_t weight_vec_1 = vld1q_f32(block_weights_1 + block_idx * BLOCK_SIZE);

      // Multiply and accumulate.
      sum_vec = vmlaq_f32(vmlaq_f32(sum_vec, weight_vec_0, input_vec_0), weight_vec_1, input_vec_1);
    }

    // Compute the horizontal sum, add in the bias, and write to the output.
    float32x2_t res = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
    float sum = vget_lane_f32(vpadd_f32(res, res), 0);
    output[output_idx] = bias[output_idx] + sum;
  }
}

}  // namespace wavernn::neon

#elif defined(__AVX512F__)
namespace wavernn::avx512 {
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
    __m512 sum_vec_0 = _mm512_set1_ps(0);
    __m512 sum_vec_1 = sum_vec_0;
    __m512 sum_vec_2 = sum_vec_0;
    __m512 sum_vec_3 = sum_vec_0;
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      // Load the index of each block of the input.
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      // Load the data for the 4 input blocks.
      __m512 input_vec_0 = _mm512_loadu_ps(input + input_idx_0);
      __m512 input_vec_1 = _mm512_loadu_ps(input + input_idx_1);
      __m512 input_vec_2 = _mm512_loadu_ps(input + input_idx_2);
      __m512 input_vec_3 = _mm512_loadu_ps(input + input_idx_3);

      // Load the data for the 4 weight blocks.
      __m512 weight_vec_0 =
          _mm512_loadu_ps(block_weights + (block_idx + 0) * BLOCK_SIZE);
      __m512 weight_vec_1 =
          _mm512_loadu_ps(block_weights + (block_idx + 2) * BLOCK_SIZE);
      __m512 weight_vec_2 =
          _mm512_loadu_ps(block_weights + (block_idx + 2) * BLOCK_SIZE);
      __m512 weight_vec_3 =
          _mm512_loadu_ps(block_weights + (block_idx + 3) * BLOCK_SIZE);

      // Perform multiplications and accumulate into the accumulators.
      sum_vec_0 = _mm512_fmadd_ps(input_vec_0, weight_vec_0, sum_vec_0);
      sum_vec_1 = _mm512_fmadd_ps(input_vec_1, weight_vec_1, sum_vec_1);
      sum_vec_2 = _mm512_fmadd_ps(input_vec_2, weight_vec_2, sum_vec_2);
      sum_vec_3 = _mm512_fmadd_ps(input_vec_3, weight_vec_3, sum_vec_3);
    }

    // Sum up accumulators to get an accumulator for the follow-up loop.
    __m512 sum_vec = _mm512_add_ps(_mm512_add_ps(sum_vec_0, sum_vec_1),
                                   _mm512_add_ps(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      // Load the input block.
      __m512 input_vec = _mm512_loadu_ps(input + block_indices[block_idx]);

      // Load the weight block.
      __m512 weight_vec =
          _mm512_loadu_ps(block_weights + block_idx * BLOCK_SIZE);

      // Multiply and accumulate.
      sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
    }

    // Compute the horizontal sum, add in the bias, and write to the output.
    output[output_idx] = bias[output_idx] + _mm512_reduce_add_ps(sum_vec);
  }
}

float quantize_vector(int size, const float* const __restrict__ input,
                      int16_t* const __restrict__ output) {
  // Magnitude must be non-zero.
  float maxMagnitude = 1.0e-3;
#pragma omp simd
  for (int i = 0; i < size; i++) {
    float value = input[i];
    float magnitude = value > 0 ? value : -value;
    maxMagnitude = magnitude > maxMagnitude ? magnitude : maxMagnitude;
  }

  float fp32ToInt16 = QUANTIZATION_SCALE / maxMagnitude;
  float int16ToFp32 = maxMagnitude / QUANTIZATION_SCALE;

  for (int i = 0; i < size; i++) {
    output[i] = int16_t(input[i] * fp32ToInt16);
  }

  return int16ToFp32;
}

inline __m512i _mm512_loadu_epi16(const void* addr) {
  return _mm512_loadu_si512((__m512*) addr);
}

/**
 * Compute a quantized block-sparse sparse matrix-vector multiply.
 *
 * @param output_size The number of output rows in this gemv.
 * @param output Where to write the output (dense float32 vector).
 * @param input Where to read the input data from (dense int16 vector).
 * @param inputScale The scale the input was divided by to convert to int16.
 * @param weights Packed weights.
 * @param bias The bias vector (dense float32 vector).
 * @param blocksPerRow Number of blocks involved in each row multiply.
 * @param rowOffsets The offset in the weights for the nth row.
 * @param indices The indices in the input vector to load blocks from.
 * @param rowScales The scales the rows were divided by to convert to int16.
 */
void sparse_gemv_quantized(int output_size, float* const __restrict__ output,
                           const int16_t* const __restrict__ input,
                           float inputScale,
                           const int16_t* const __restrict__ weights,
                           const float* const __restrict__ bias,
                           const int* const __restrict__ blocksPerRow,
                           const int* const __restrict__ rowOffsets,
                           const int* const __restrict__ indices,
                           const float* const __restrict__ rowScales) {
  // Parallelize the computation across the output row. Each thread can
  // separately compute outputs for a block of the output vector.
#pragma omp parallel for
  for (int output_idx = 0; output_idx < output_size; output_idx++) {
    // Number of block multiplications for this row.
    int n_blocks = blocksPerRow[output_idx];

    // Where the weights for this row start.
    const int16_t* block_weights = weights + rowOffsets[output_idx];

    // Where the input indices for this row start.
    const int* block_indices =
        indices + rowOffsets[output_idx] / QUANTIZED_BLOCK_SIZE;

    // How much to scale final output by.
    float outputScale = inputScale * rowScales[output_idx];

    // Unroll the block multiplication loop by 4. This speeds up the gemvs by a
    // little bit. If the number of blocks is not a multiple of 4, the follow-up
    // loop will take care of it.
    int block_idx = 0;
    int n_blocks_grouped = n_blocks - (n_blocks % 4);
    __m512i sum_vec_0 = _mm512_set1_epi32(0);
    __m512i sum_vec_1 = sum_vec_0;
    __m512i sum_vec_2 = sum_vec_0;
    __m512i sum_vec_3 = sum_vec_0;
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      // Load the index of each block of the input.
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      // Load the data for the 4 input blocks.
      __m512i input_vec_0 = _mm512_loadu_epi16(input + input_idx_0);
      __m512i input_vec_1 = _mm512_loadu_epi16(input + input_idx_1);
      __m512i input_vec_2 = _mm512_loadu_epi16(input + input_idx_2);
      __m512i input_vec_3 = _mm512_loadu_epi16(input + input_idx_3);

      // Load the data for the 4 weight blocks.
      __m512i weight_vec_0 = _mm512_loadu_epi16(
          block_weights + (block_idx + 0) * QUANTIZED_BLOCK_SIZE);
      __m512i weight_vec_1 = _mm512_loadu_epi16(
          block_weights + (block_idx + 1) * QUANTIZED_BLOCK_SIZE);
      __m512i weight_vec_2 = _mm512_loadu_epi16(
          block_weights + (block_idx + 2) * QUANTIZED_BLOCK_SIZE);
      __m512i weight_vec_3 = _mm512_loadu_epi16(
          block_weights + (block_idx + 3) * QUANTIZED_BLOCK_SIZE);

      // Perform multiplications and accumulate into the accumulators.
      sum_vec_0 = _mm512_add_epi32(_mm512_madd_epi16(input_vec_0, weight_vec_0),
                                   sum_vec_0);
      sum_vec_1 = _mm512_add_epi32(_mm512_madd_epi16(input_vec_1, weight_vec_1),
                                   sum_vec_1);
      sum_vec_2 = _mm512_add_epi32(_mm512_madd_epi16(input_vec_2, weight_vec_2),
                                   sum_vec_2);
      sum_vec_3 = _mm512_add_epi32(_mm512_madd_epi16(input_vec_3, weight_vec_3),
                                   sum_vec_3);
    }

    // Sum up accumulators to get an accumulator for the follow-up loop.
    __m512i sum_vec = _mm512_add_epi32(_mm512_add_epi32(sum_vec_0, sum_vec_1),
                                       _mm512_add_epi32(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      // Load the input block.
      __m512i input_vec =
          _mm512_loadu_epi16(input + block_indices[block_idx]);

      // Load the weight block.
      __m512i weight_vec = _mm512_loadu_epi16(
          block_weights + block_idx * QUANTIZED_BLOCK_SIZE);

      // Multiply and accumulate.
      sum_vec =
          _mm512_add_epi32(sum_vec, _mm512_madd_epi16(input_vec, weight_vec));
    }

    __m512 float_vec = _mm512_cvtepi32_ps(sum_vec);
    float_vec = _mm512_mul_ps(float_vec, _mm512_set1_ps(outputScale));

    // Compute the horizontal sum, add in the bias, and write to the output.
    output[output_idx] = bias[output_idx] + _mm512_reduce_add_ps(float_vec);
  }
}

}  // namespace wavernn::avx512
#elif defined(__AVX2__)
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

float quantize_vector(int size, const float* const __restrict__ input,
                      int16_t* const __restrict__ output) {
  // Magnitude must be non-zero.
  float maxMagnitude = 1.0e-3;
#pragma omp simd
  for (int i = 0; i < size; i++) {
    float value = input[i];
    float magnitude = value > 0 ? value : -value;
    maxMagnitude = magnitude > maxMagnitude ? magnitude : maxMagnitude;
  }

  float fp32ToInt16 = QUANTIZATION_SCALE / maxMagnitude;
  float int16ToFp32 = maxMagnitude / QUANTIZATION_SCALE;

  for (int i = 0; i < size; i++) {
    output[i] = int16_t(input[i] * fp32ToInt16);
  }

  return int16ToFp32;
}

/**
 * Compute a quantized block-sparse sparse matrix-vector multiply.
 *
 * @param output_size The number of output rows in this gemv.
 * @param output Where to write the output (dense float32 vector).
 * @param input Where to read the input data from (dense int16 vector).
 * @param inputScale The scale the input was divided by to convert to int16.
 * @param weights Packed weights.
 * @param bias The bias vector (dense float32 vector).
 * @param blocksPerRow Number of blocks involved in each row multiply.
 * @param rowOffsets The offset in the weights for the nth row.
 * @param indices The indices in the input vector to load blocks from.
 * @param rowScales The scales the rows were divided by to convert to int16.
 */
void sparse_gemv_quantized(int output_size, float* const __restrict__ output,
                           const int16_t* const __restrict__ input,
                           float inputScale,
                           const int16_t* const __restrict__ weights,
                           const float* const __restrict__ bias,
                           const int* const __restrict__ blocksPerRow,
                           const int* const __restrict__ rowOffsets,
                           const int* const __restrict__ indices,
                           const float* const __restrict__ rowScales) {
  // Parallelize the computation across the output row. Each thread can
  // separately compute outputs for a block of the output vector.
#pragma omp parallel for
  for (int output_idx = 0; output_idx < output_size; output_idx++) {
    // Number of block multiplications for this row.
    int n_blocks = blocksPerRow[output_idx];

    // Where the weights for this row start.
    const int16_t* block_weights = weights + rowOffsets[output_idx];

    // Where the input indices for this row start.
    const int* block_indices =
        indices + rowOffsets[output_idx] / QUANTIZED_BLOCK_SIZE;

    // How much to scale final output by.
    float outputScale = inputScale * rowScales[output_idx];

    // Unroll the block multiplication loop by 4. This speeds up the gemvs by a
    // little bit. If the number of blocks is not a multiple of 4, the follow-up
    // loop will take care of it.
    int block_idx = 0;
    int n_blocks_grouped = n_blocks - (n_blocks % 4);
    __m256i sum_vec_0 = _mm256_set1_epi32(0);
    __m256i sum_vec_1 = sum_vec_0;
    __m256i sum_vec_2 = sum_vec_0;
    __m256i sum_vec_3 = sum_vec_0;
    for (; block_idx < n_blocks_grouped; block_idx += 4) {
      // Load the index of each block of the input.
      int input_idx_0 = block_indices[block_idx + 0];
      int input_idx_1 = block_indices[block_idx + 1];
      int input_idx_2 = block_indices[block_idx + 2];
      int input_idx_3 = block_indices[block_idx + 3];

      // Load the data for the 4 input blocks.
      __m256i input_vec_0 = _mm256_loadu_si256((__m256i*)(input + input_idx_0));
      __m256i input_vec_1 = _mm256_loadu_si256((__m256i*)(input + input_idx_1));
      __m256i input_vec_2 = _mm256_loadu_si256((__m256i*)(input + input_idx_2));
      __m256i input_vec_3 = _mm256_loadu_si256((__m256i*)(input + input_idx_3));

      // Load the data for the 4 weight blocks.
      __m256i weight_vec_0 = _mm256_loadu_si256(
          (__m256i*)(block_weights + (block_idx + 0) * QUANTIZED_BLOCK_SIZE));
      __m256i weight_vec_1 = _mm256_loadu_si256(
          (__m256i*)(block_weights + (block_idx + 1) * QUANTIZED_BLOCK_SIZE));
      __m256i weight_vec_2 = _mm256_loadu_si256(
          (__m256i*)(block_weights + (block_idx + 2) * QUANTIZED_BLOCK_SIZE));
      __m256i weight_vec_3 = _mm256_loadu_si256(
          (__m256i*)(block_weights + (block_idx + 3) * QUANTIZED_BLOCK_SIZE));

      // Perform multiplications and accumulate into the accumulators.
      sum_vec_0 = _mm256_add_epi32(_mm256_madd_epi16(input_vec_0, weight_vec_0),
                                   sum_vec_0);
      sum_vec_1 = _mm256_add_epi32(_mm256_madd_epi16(input_vec_1, weight_vec_1),
                                   sum_vec_1);
      sum_vec_2 = _mm256_add_epi32(_mm256_madd_epi16(input_vec_2, weight_vec_2),
                                   sum_vec_2);
      sum_vec_3 = _mm256_add_epi32(_mm256_madd_epi16(input_vec_3, weight_vec_3),
                                   sum_vec_3);
    }

    // Sum up accumulators to get an accumulator for the follow-up loop.
    __m256i sum_vec = _mm256_add_epi32(_mm256_add_epi32(sum_vec_0, sum_vec_1),
                                       _mm256_add_epi32(sum_vec_2, sum_vec_3));
    for (; block_idx < n_blocks; block_idx++) {
      // Load the input block.
      __m256i input_vec =
          _mm256_loadu_si256((__m256i*)(input + block_indices[block_idx]));

      // Load the weight block.
      __m256i weight_vec = _mm256_loadu_si256(
          (__m256i*)(block_weights + block_idx * QUANTIZED_BLOCK_SIZE));

      // Multiply and accumulate.
      sum_vec =
          _mm256_add_epi32(sum_vec, _mm256_madd_epi16(input_vec, weight_vec));
    }

    __m256 float_vec = _mm256_cvtepi32_ps(sum_vec);
    float_vec = _mm256_mul_ps(float_vec, _mm256_set1_ps(outputScale));

    // Compute the horizontal sum, add in the bias, and write to the output.
    output[output_idx] = bias[output_idx] + mm256_horizontal_sum(float_vec);
  }
}

}  // namespace wavernn::avx2
#endif

namespace wavernn {

PackedLinear::PackedLinear(const Tensor& matrix, const Tensor& bias,
                           bool quantized)
    : quantized_(quantized), matrix_(matrix), bias_(bias) {
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

  // Repack the matrix for quantization. This repacking is similar to
  // floating-point repacking, but uses larger block sizes, converts the data
  // to integer format, and computes per-row scaling factors.
  // matrix in blocks and check if the block is nonzero. If the block is
  // nonzero, then add its weights to the linearly growing weight vector, and
  // record the location in the input where we would multiply this by. Record
  // the number of blocks needed for each row.
  for (int r = 0; r < output_size_; r++) {
    quantizedRowOffsets_.push_back(quantizedData_.size());

    // Find the maximum magnitude in this row.
    float maxWeightMagnitude = 0.0f;
    for (int c = 0; c < input_size_; c++) {
      maxWeightMagnitude = std::max(maxWeightMagnitude, std::abs(mat[r][c]));
    }
    float fp32ToInt16 = QUANTIZATION_SCALE / maxWeightMagnitude;
    float int16ToFp32 = maxWeightMagnitude / QUANTIZATION_SCALE;

    int numBlocks = 0;
    for (int c = 0; c < input_size_; c += QUANTIZED_BLOCK_SIZE) {
      bool empty = true;
      for (int i = c; i < c + QUANTIZED_BLOCK_SIZE; i++) {
        if (mat[r][i] != 0) {
          empty = false;
          break;
        }
      }

      if (!empty) {
        numBlocks++;
        quantizedIndices_.push_back(c);
        for (int i = c; i < c + QUANTIZED_BLOCK_SIZE; i++) {
          quantizedData_.push_back(int16_t(mat[r][i] * fp32ToInt16));
        }
      }
    }

    quantizedBlocksPerRow_.push_back(numBlocks);
    quantizedRowScales_.push_back(int16ToFp32);
  }
}

void PackedLinear::gemv(Tensor& out, const Tensor& vector) const {
  ASSERT_TENSOR_SIZE(out, output_size_);
  ASSERT_TENSOR_SIZE(vector, input_size_);

  // Quantized inference only available on some platforms.
#if defined(__AVX2__) || defined(__AVX512F__)
  bool allowQuantized = true;
#else
  bool allowQuantized = false;
#endif

  if (allowQuantized && quantized_) {

#if defined(__AVX2__) || defined(__AVX512F__)
    const float* const __restrict__ input = vector.data_ptr<float>();
    float* const __restrict__ output = out.data_ptr<float>();
    const int16_t* const __restrict__ weights = quantizedData_.data();
    const float* const __restrict__ bias = bias_.data_ptr<float>();
    const int* const __restrict__ blocksPerRow = quantizedBlocksPerRow_.data();
    const int* const __restrict__ rowOffsets = quantizedRowOffsets_.data();
    const int* const __restrict__ indices = quantizedIndices_.data();
    const float* const __restrict__ rowScales = quantizedRowScales_.data();

    int16_t inputQuantized[input_size_];
#endif

#if defined(__AVX512F__)
    float inputScale =
        wavernn::avx512::quantize_vector(input_size_, input, inputQuantized);
    wavernn::avx512::sparse_gemv_quantized(
        output_size_, output, inputQuantized, inputScale, weights, bias,
        blocksPerRow, rowOffsets, indices, rowScales);
#elif defined(__AVX2__)
    float inputScale =
        wavernn::avx2::quantize_vector(input_size_, input, inputQuantized);
    wavernn::avx2::sparse_gemv_quantized(
        output_size_, output, inputQuantized, inputScale, weights, bias,
        blocksPerRow, rowOffsets, indices, rowScales);
#else
    throw std::runtime_error("Quantized inference only supported with AVX");
#endif
  } else {
    float* const __restrict__ output = out.data_ptr<float>();
    const float* const __restrict__ input = vector.data_ptr<float>();
    const float* const __restrict__ weights = data_.data();
    const float* const __restrict__ bias = bias_.data_ptr<float>();
    const int* const __restrict__ blocksPerRow = blocksPerRow_.data();
    const int* const __restrict__ rowOffsets = rowOffsets_.data();
    const int* const __restrict__ indices = indices_.data();

#if defined(__AVX512F__)
    wavernn::avx512::sparse_gemv(output_size_, output, input, weights, bias,
                                 blocksPerRow, rowOffsets, indices);
#elif defined(__AVX2__)
    wavernn::avx2::sparse_gemv(output_size_, output, input, weights, bias,
                               blocksPerRow, rowOffsets, indices);
#elif defined(__ARM_NEON)
    wavernn::neon::sparse_gemv(output_size_, output, input, weights, bias,
                               blocksPerRow, rowOffsets, indices);
#else
    wavernn::fallback::gemv(out, bias_, matrix_, vector);
#endif
  }
}

}  // namespace wavernn
