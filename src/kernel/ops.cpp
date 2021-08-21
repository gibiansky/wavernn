#include "ops.h"

#if defined(__AVX2__)
#include <immintrin.h>
#include <mkl.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <torch/script.h>

#include <random>
#include <vector>

#include "wavernn_assert.h"

using torch::Tensor;

namespace {

/**
 * Sample from a categorical distribution defined by the input probabilities.
 *
 * @param distribution Probabilities which sum to one of shape [num_classes]..
 * @returns A sampled discrete value in the range [0, num_classes - 1].
 */
int SampleFromSoftmax(const Tensor &distribution) {
  // In order to make this function incredibly fast (since it may be in a tight
  // inner loop), we will precompute a large quantity of random values and cycle
  // through them. If the amount of random values is large enough, this will not
  // be audible in the generated audio. The random values are populated from a
  // random number generator the first time this function is called, so the very
  // first call to this function will be much, much slower than all the
  // subsequent calls.

  // A static buffer of random values to cycle through.
  static std::vector<float> random;

  // The current index of the next random value to generate.
  static size_t idx = 0;

  // Fill the random value buffer the first time this is used.
  if (random.empty()) {
    random.resize(50000);

    std::default_random_engine generator;  // NOLINT
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (float &flt : random) {
      flt = distribution(generator);
    }
  }

  // Select the next value from the array of random values and advance to the
  // next index, wrapping around if necessary.
  float rand = random[idx];
  idx = (idx + 1) % random.size();

  // Compute the running sum of the distribution and find the first index at
  // which the running sum is larger than the generated random value.
  float sum = 0.0;
  const float *dist = distribution.data_ptr<float>();
  int dim = distribution.size(0);
  for (int i = 0; i < dim; i++) {
    sum += dist[i];
    if (sum >= rand) {
      return i;
    }
  }

  // It's possible (albeit unlikely) that due to floating point error, the sum
  // of the distribution is less than the generated random value. In this case,
  // return num_classes - 1 as the sample.
  return dim - 1;
}

/**
 * Sample from a categorical distribution defined by the pre-softmax logits.
 *
 * This is done using the Gumbel-Softmax trick.
 *
 * @param distribution Logits of shape [num_classes]..
 * @returns A sampled discrete value in the range [0, num_classes - 1].
 */
int SampleFromLogits(const Tensor &logits) {
  // In order to make this function incredibly fast (since it may be in a tight
  // inner loop), we will precompute a large quantity of random values and cycle
  // through them. If the amount of random values is large enough, this will not
  // be audible in the generated audio. The random values are populated from a
  // random number generator the first time this function is called, so the very
  // first call to this function will be much, much slower than all the
  // subsequent calls.

  // A static buffer of random values to cycle through.
  static std::vector<float> random;

  // The current index of the next random value to generate.
  static size_t idx = 0;

  // Fill the random value buffer the first time this is used.
  if (random.empty()) {
    random.resize(256 * 50000);

    std::default_random_engine generator;  // NOLINT
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (float &flt : random) {
      flt = -std::log(-std::log(distribution(generator)));
    }
  }

  const float *lgts = logits.data_ptr<float>();
  const float *randoms = random.data();
  int dim = logits.size(0);

  // Reset to start if needed.
  if (idx + dim >= random.size()) {
    idx = 0;
  }

  float maxVal = -9999.9f;
  int maxIdx = 0;
  for (int i = 0; i < dim; i++) {
    float val = lgts[i] + randoms[idx + i];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }

  idx += dim;
  return maxIdx;
}

}  // namespace

namespace wavernn {

#if defined(__AVX512F__)

namespace avx512 {
// Source:
// https://pdfs.semanticscholar.org/bb2a/f84f8a179ac5486cf197c409c01289ff9064.pdf
// https://github.com/hfp/libxsmm/blob/55c6a9f92a6ff0b7124ff351aa3f7c20ec789170/include/libxsmm_intrinsics_x86.h#L653
inline __m512 _mm512_tanh_ps(__m512 x) {
  const __m512 c0 = _mm512_set1_ps(2027025.0f);
  const __m512 c1 = _mm512_set1_ps(270270.0f);
  const __m512 c2 = _mm512_set1_ps(6930.0f);
  const __m512 c3 = _mm512_set1_ps(36.0f);
  const __m512 c1_d = _mm512_set1_ps(945945.0f);
  const __m512 c2_d = _mm512_set1_ps(51975.0f);
  const __m512 c3_d = _mm512_set1_ps(630.0f);
  const __m512 hi_bound = _mm512_set1_ps(4.97f);
  const __m512 lo_bound = _mm512_set1_ps(-4.97f);
  const __m512 ones = _mm512_set1_ps(1.0f);
  const __m512 neg_ones = _mm512_set1_ps(-1.0f);

  const __m512 x2 = _mm512_mul_ps(x, x);
  const __m512 t1_nom = _mm512_fmadd_ps(c3, x2, c2);
  const __m512 t2_nom = _mm512_fmadd_ps(t1_nom, x2, c1);
  const __m512 t3_nom = _mm512_fmadd_ps(t2_nom, x2, c0);
  const __m512 nom = _mm512_mul_ps(t3_nom, x);
  const __m512 t1_denom = _mm512_add_ps(x2, c3_d);
  const __m512 t2_denom = _mm512_fmadd_ps(t1_denom, x2, c2_d);
  const __m512 t3_denom = _mm512_fmadd_ps(t2_denom, x2, c1_d);
  const __m512 denom = _mm512_fmadd_ps(t3_denom, x2, c0);
  const __m512 denom_rcp = _mm512_rcp14_ps(denom);
  const __mmask16 mask_hi = _mm512_cmp_ps_mask(x, hi_bound, _CMP_GT_OQ);
  const __mmask16 mask_lo = _mm512_cmp_ps_mask(x, lo_bound, _CMP_LT_OQ);
  __m512 result = _mm512_mul_ps(nom, denom_rcp);
  result = _mm512_mask_blend_ps(mask_hi, result, ones);
  result = _mm512_mask_blend_ps(mask_lo, result, neg_ones);

  return result;
}

inline __m512 _mm512_sigmoid_ps(__m512 x) {
  const __m512 half = _mm512_set1_ps(0.5f);
  x = _mm512_mul_ps(x, half);
  x = _mm512_tanh_ps(x);
  return _mm512_fmadd_ps(half, x, half);
}

}  // namespace avx512
#elif defined(__AVX2__)
namespace avx2 {
inline __m256 _mm256_tanh_ps(__m256 x) {
  const __m256 c0 = _mm256_set1_ps(2027025.0f);
  const __m256 c1 = _mm256_set1_ps(270270.0f);
  const __m256 c2 = _mm256_set1_ps(6930.0f);
  const __m256 c3 = _mm256_set1_ps(36.0f);
  const __m256 c1_d = _mm256_set1_ps(945945.0f);
  const __m256 c2_d = _mm256_set1_ps(51975.0f);
  const __m256 c3_d = _mm256_set1_ps(630.0f);
  const __m256 hi_bound = _mm256_set1_ps(4.97f);
  const __m256 lo_bound = _mm256_set1_ps(-4.97f);
  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256 neg_ones = _mm256_set1_ps(-1.0f);

  const __m256 x2 = _mm256_mul_ps(x, x);
  const __m256 t1_nom = _mm256_fmadd_ps(c3, x2, c2);
  const __m256 t2_nom = _mm256_fmadd_ps(t1_nom, x2, c1);
  const __m256 t3_nom = _mm256_fmadd_ps(t2_nom, x2, c0);
  const __m256 nom = _mm256_mul_ps(t3_nom, x);
  const __m256 t1_denom = _mm256_add_ps(x2, c3_d);
  const __m256 t2_denom = _mm256_fmadd_ps(t1_denom, x2, c2_d);
  const __m256 t3_denom = _mm256_fmadd_ps(t2_denom, x2, c1_d);
  const __m256 denom = _mm256_fmadd_ps(t3_denom, x2, c0);
  const __m256 denom_rcp = _mm256_rcp_ps(denom);
  const __m256 mask_hi = _mm256_cmp_ps(x, hi_bound, _CMP_GT_OQ);
  const __m256 mask_lo = _mm256_cmp_ps(x, lo_bound, _CMP_LT_OQ);
  __m256 result = _mm256_mul_ps(nom, denom_rcp);
  result = _mm256_blendv_ps(result, ones, mask_hi);
  result = _mm256_blendv_ps(result, neg_ones, mask_lo);

  return result;
}

inline __m256 _mm256_sigmoid_ps(__m256 x) {
  const __m256 half = _mm256_set1_ps(0.5f);
  x = _mm256_mul_ps(x, half);
  x = _mm256_tanh_ps(x);
  return _mm256_fmadd_ps(half, x, half);
}

}  // namespace avx2

#endif

int SoftmaxSampleFromLogits(const torch::Tensor &logits) {
  // To sample without the Gumbel-Softmax trick...
  //
  //     auto distribution = at::softmax(logits, 0);
  //     return SampleFromSoftmax(distribution);
  //
  // Sampling using Gumbel-Softmax trick is faster.
  return SampleFromLogits(logits);
}

void UpdateGruState(const torch::Tensor &gruState, const torch::Tensor &gruIh,
                    const torch::Tensor &gruHh) {
  const int size = gruState.size(0);
  float *ih = gruIh.data_ptr<float>();
  float *hh = gruHh.data_ptr<float>();
  float *state = gruState.data_ptr<float>();

#if defined(__AVX2__)
  // AVX2 implementation.
  ASSERT_BOOL(size % 8 == 0);

  for (int i = 0; i < size; i += 8) {
    __m256 ih_r = _mm256_loadu_ps(ih + i);
    __m256 hh_r = _mm256_loadu_ps(hh + i);
    __m256 r = avx2::_mm256_sigmoid_ps(_mm256_add_ps(ih_r, hh_r));

    __m256 ih_z = _mm256_loadu_ps(ih + size + i);
    __m256 hh_z = _mm256_loadu_ps(hh + size + i);
    __m256 z = avx2::_mm256_sigmoid_ps(_mm256_add_ps(ih_z, hh_z));

    __m256 ih_n = _mm256_loadu_ps(ih + 2 * size + i);
    __m256 hh_n = _mm256_loadu_ps(hh + 2 * size + i);
    __m256 n = avx2::_mm256_tanh_ps(_mm256_fmadd_ps(r, hh_n, ih_n));

    __m256 z1m = _mm256_sub_ps(_mm256_set1_ps(1.0), z);
    __m256 s = _mm256_loadu_ps(state + i);
    __m256 s_new = _mm256_add_ps(_mm256_mul_ps(z1m, n), _mm256_mul_ps(s, z));
    _mm256_storeu_ps(state + i, s_new);
  }
#else
  // Fallback implementation.
#pragma omp parallel for
  for (int i = 0; i < 2 * size; i++) {
    ih[i] = 1.0 / (1.0f + std::exp(-ih[i] - hh[i]));
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    ih[2 * size + i] = std::tanh(ih[i] * ih[2 * size + i] + hh[2 * size + i]);
  }

  float *z = ih + size;
  float *n = ih + 2 * size;
#pragma omp simd
  for (int i = 0; i < size; i++) {
    state[i] = (1 - z[i]) * n[i] + z[i] * state[i];
  }

#endif
}

/**
 * Compute the WaveRNN GRU layer input vector. The input vector is the sum of
 * the sample embedding and the conditioner output corresponding to the current
 * spectrogram frame. The sample embedding for a discrete sample value k is the
 * kth row of the sample embedding matrix; replaced with zeros.
 *
 * @param output Where to write the output to.
 * @param sampleEmbeddings The matrix of sample embeddings.
 * @param conditionerOutputs The matrix of conditioner outputs.
 * @param sample The last sample produced.
 * @param frameIndex The index of the current frame. Zero corresponds to the
 * first conditioner output.
 */
void GruInput(const torch::Tensor &output,
              const torch::Tensor &sampleEmbeddings,
              const torch::Tensor &conditionerOutputs, int sample,
              int frameIndex) {
  int size = output.size(0);
  float *sampleEmbeddingsPtr = sampleEmbeddings.data_ptr<float>();
  float *conditionerOutputPtr = conditionerOutputs.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();

#if defined(__AVX2__)
  vsAdd(size, sampleEmbeddingsPtr + sample * size,
        conditionerOutputPtr + frameIndex * size, output_ptr);
#else
  ASSERT_BOOL(size % 4 == 0);
  float *sampleEmbeds = sampleEmbeddingsPtr + sample * size;
  float *condOuts = conditionerOutputPtr + frameIndex * size;
  for (int i = 0; i < size; i += 4) {
    vst1q_f32(output_ptr + i,
              vaddq_f32(vld1q_f32(sampleEmbeds + i), vld1q_f32(condOuts + i)));
  }
#endif
}

/**
 * In-place element-wise ReLU. All negative elements in the input tensor are
 * replaced with zeros.
 *
 * @param values Values to apply ReLU operation to.
 */
void ReLU(const torch::Tensor &tensor) {
  float *h = tensor.data_ptr<float>();
  int size = tensor.size(0);
  for (int i = 0; i < size; i++) {
    h[i] = h[i] < 0 ? 0.0f : h[i];
  }
}

}  // namespace wavernn
