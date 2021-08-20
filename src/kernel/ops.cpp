#include "ops.h"

#include <immintrin.h>
#include <mkl.h>
#include <torch/script.h>

#include <random>
#include <vector>

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

}  // namespace

namespace wavernn {

int SoftmaxSampleFromLogits(const torch::Tensor &logits) {
  auto distribution = at::softmax(logits, 0);
  return SampleFromSoftmax(distribution);
}

void UpdateGruState(const torch::Tensor &gruState, const torch::Tensor &gruIh,
                    const torch::Tensor &gruHh) {
  const int size = gruState.size(0);
  float *ih = gruIh.data_ptr<float>();
  float *hh = gruHh.data_ptr<float>();
  float *state = gruState.data_ptr<float>();

  vsAdd(2 * size, ih, hh, ih);

#pragma omp simd
  for (int i = 0; i < 2 * size; i++) {
    ih[i] = -ih[i];
  }

  vsExp(2 * size, ih, ih);

#pragma omp simd
  for (int i = 0; i < 2 * size; i++) {
    ih[i]++;
  }

  // 14%
  vsInv(2 * size, ih, ih);
  vsMul(size, ih, hh + 2 * size, hh + 2 * size);
  vsAdd(size, ih + 2 * size, hh + 2 * size, ih + 2 * size);
  vsTanh(size, ih + 2 * size, ih + 2 * size);

  // 1%
  float *z = ih + size;
  float *n = ih + 2 * size;
#pragma omp simd
  for (int i = 0; i < size; i++) {
    state[i] = (1 - z[i]) * n[i] + z[i] * state[i];
  }
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

  vsAdd(size, sampleEmbeddingsPtr + sample * size,
        conditionerOutputPtr + frameIndex * size, output_ptr);
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
