#include <torch/script.h>

#include <chrono>
#include <vector>

#include "gemv.h"
#include "ops.h"
#include "timer.h"
#include "wavernn_assert.h"

namespace wavernn {

using torch::Tensor;

Tensor wavernn_inference(
    /* WaveRNN inputs */
    const Tensor &gru_activations_ih, const Tensor &previous_sample,
    const Tensor &gru_state,

    /* WaveRNN weights */
    const std::vector<Tensor> &sample_embeddings, const Tensor &gru_weights_hh,
    const Tensor &gru_bias_hh, const Tensor &hidden_weights,
    const Tensor &hidden_bias, const std::vector<Tensor> &output_weights,
    const std::vector<Tensor> &output_biases,

    /* WaveRNN hyperparameters */
    const int64_t hop_length, const bool timing) {
  Timer timer(timing);
  timer.start("Initialization");
  c10::InferenceMode inferenceMode;

  // Extract hyperparameters.
  ASSERT_BOOL(!sample_embeddings.empty());
  auto num_bands = (long)sample_embeddings.size();
  auto num_frames = gru_activations_ih.size(0);
  auto gru_state_size = gru_state.size(0);
  auto hidden_size = hidden_bias.size(0);
  auto output_size = output_biases[0].size(0);

  // Verify sizes.
  ASSERT_BOOL((int)sample_embeddings.size() == num_bands);
  ASSERT_BOOL((int)output_weights.size() == num_bands);
  ASSERT_BOOL((int)output_biases.size() == num_bands);
  ASSERT_TENSOR_SIZE(gru_activations_ih, num_frames, 3 * gru_state_size);
  ASSERT_TENSOR_SIZE(previous_sample, num_bands);
  ASSERT_TENSOR_SIZE(gru_state, gru_state_size);
  ASSERT_TENSOR_SIZE(gru_weights_hh, 3 * gru_state_size, gru_state_size);
  ASSERT_TENSOR_SIZE(gru_bias_hh, 3 * gru_state_size);
  ASSERT_TENSOR_SIZE(hidden_weights, hidden_size, gru_state_size);
  ASSERT_TENSOR_SIZE(hidden_bias, hidden_size);

  for (int i = 0; i < num_bands; i++) {
    ASSERT_TENSOR_SIZE(sample_embeddings[i], output_size, 3 * gru_state_size);
    ASSERT_TENSOR_SIZE(output_weights[i], output_size, hidden_size);
    ASSERT_TENSOR_SIZE(output_biases[i], output_size);
  }

  // Allocate temporary and output buffers.
  Tensor gru_input = torch::zeros(3 * gru_state_size);
  Tensor gru_activations_hh = torch::zeros(3 * gru_state_size);
  Tensor hidden = torch::zeros(hidden_size);
  Tensor output_tensor =
      torch::zeros({num_frames * hop_length, num_bands},
                   torch::TensorOptions().dtype(torch::kInt64));
  auto outputs = output_tensor.accessor<long, 2>();

  PackedLinear gruMat(gru_weights_hh, gru_bias_hh);
  PackedLinear hiddenMat(hidden_weights, hidden_bias);

  std::vector<PackedLinear> outputMats;
  std::vector<Tensor> logits;
  for (auto i = 0; i < num_bands; i++) {
    outputMats.emplace_back(output_weights[i], output_biases[i]);
    logits.emplace_back(torch::zeros(output_size));
  }

  long *samples = previous_sample.data_ptr<long>();

  for (int timestep = 0; timestep < outputs.size(0); timestep++) {
    int frame_idx = timestep / int(hop_length);
    timer.start("GRU Input");
    GruInput(gru_input, sample_embeddings, gru_activations_ih, samples,
             frame_idx);

    timer.start("GRU GEMV");
    gruMat.gemv(gru_activations_hh, gru_state);

    timer.start("GRU Nonlinearities");
    UpdateGruState(gru_state, gru_input, gru_activations_hh);

    timer.start("Hidden GEMV");
    hiddenMat.gemv(hidden, gru_state);

    timer.start("Hidden ReLU");
    ReLU(hidden);

    timer.start("Output GEMV");
    for (auto i = 0; i < num_bands; i++) {
      outputMats[i].gemv(logits[i], hidden);
    }

    timer.start("SampleSoftmax");
    for (auto i = 0; i < num_bands; i++) {
      samples[i] = SoftmaxSampleFromLogits(logits[i]);
      outputs[timestep][i] = samples[i];
    }
  }

  timer.print();

  return output_tensor;
}

Tensor sparse_gemv(const Tensor &bias, const Tensor &matrix,
                   const Tensor &vector) {
  c10::InferenceMode inferenceMode;
  PackedLinear mat(matrix, bias);
  Tensor output = torch::zeros(bias.size(0));
  mat.gemv(output, vector);
  return output;
}

std::vector<Tensor> sparse_gemv_benchmark(const Tensor &bias,
                                          const Tensor &matrix,
                                          const Tensor &vector,
                                          int64_t warmup_iters,
                                          int64_t bench_iters) {
  c10::InferenceMode inferenceMode;
  Tensor elapsed_dense = torch::zeros(bench_iters);
  Tensor elapsed_sparse = torch::zeros(bench_iters);
  Tensor tmp = torch::zeros(bias.size(0));
  PackedLinear mat(matrix, bias);

  auto elapsed_dense_a = elapsed_dense.accessor<float, 1>();
  auto elapsed_sparse_a = elapsed_sparse.accessor<float, 1>();

  for (int i = 0; i < warmup_iters; i++) {
    torch::addmv_out(tmp, bias, matrix, vector);
  }

  for (int i = 0; i < bench_iters; i++) {
    auto start = std::chrono::steady_clock::now();
    torch::addmv_out(tmp, bias, matrix, vector);
    auto end = std::chrono::steady_clock::now();
    auto elapsedUs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    elapsed_dense_a[i] = float(elapsedUs / 1000.0);
  }

  for (int i = 0; i < warmup_iters; i++) {
    mat.gemv(tmp, vector);
  }

  for (int i = 0; i < bench_iters; i++) {
    auto start = std::chrono::steady_clock::now();
    mat.gemv(tmp, vector);
    auto end = std::chrono::steady_clock::now();
    auto elapsedUs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    elapsed_sparse_a[i] = float(elapsedUs / 1000.0);
  }

  return {elapsed_dense, elapsed_sparse};
}

}  // namespace wavernn

TORCH_LIBRARY(wavernn, m) {  // NOLINT
  m.def("wavernn_inference", &wavernn::wavernn_inference);
  m.def("sparse_gemv", &wavernn::sparse_gemv);
  m.def("sparse_gemv_benchmark", &wavernn::sparse_gemv_benchmark);
}
