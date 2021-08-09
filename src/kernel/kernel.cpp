#include <torch/script.h>

using torch::Tensor;

inline float sigmoid(float x) { return 1.0f / (1.0 + std::exp(-x)); }

void gru_nonlinearity(Tensor &gru_state, const Tensor &gru_activations_ih,
                      const Tensor &gru_activations_hh) {
  auto state = gru_state.accessor<float, 1>();
  const auto ih = gru_activations_ih.accessor<float, 1>();
  const auto hh = gru_activations_hh.accessor<float, 1>();

  const int size = state.size(0);
  for (int i = 0; i < size; i++) {
    float r = sigmoid(ih[i] + hh[i]);
    float z = sigmoid(ih[size + i] + hh[size + i]);
    float n = std::tanh(ih[2 * size + i] + r * hh[2 * size + i]);
    state[i] = (1 - z) * n + z * state[i];
  }
}

Tensor wavernn_inference(
    /* WaveRNN inputs */
    const Tensor &gru_activations_ih, Tensor &previous_sample, Tensor &gru_state,

    /* WaveRNN weights */
    const Tensor &sample_embeddings,
    const Tensor &gru_weights_hh,
    const Tensor &gru_bias_hh, const Tensor &hidden_weights,
    const Tensor &hidden_bias, const Tensor &output_weights,
    const Tensor &output_bias,

    /* WaveRNN hyperparameters */
    const int64_t hop_length) {
  c10::InferenceMode inferenceMode;

  // Extract hyperparameters.
  auto num_frames = gru_activations_ih.size(0);
  auto gru_state_size = gru_state.size(0);
  auto hidden_size = hidden_bias.size(0);
  auto output_size = output_bias.size(0);

  // Verify sizes.
  CHECK((gru_activations_ih.sizes() == c10::IntArrayRef{num_frames, 3 * gru_state_size}));
  CHECK((previous_sample.sizes() == c10::IntArrayRef{1}));
  CHECK((gru_state.sizes() == c10::IntArrayRef{gru_state_size}));
  CHECK((sample_embeddings.sizes() ==
         c10::IntArrayRef{output_size, 3 * gru_state_size}));
  CHECK((gru_weights_hh.sizes() ==
         c10::IntArrayRef{3 * gru_state_size, gru_state_size}));
  CHECK((gru_bias_hh.sizes() == c10::IntArrayRef{3 * gru_state_size}));
  CHECK((hidden_weights.sizes() ==
         c10::IntArrayRef{hidden_size, gru_state_size}));
  CHECK((hidden_bias.sizes() == c10::IntArrayRef{hidden_size}));
  CHECK((output_weights.sizes() == c10::IntArrayRef{output_size, hidden_size}));
  CHECK((output_bias.sizes() == c10::IntArrayRef{output_size}));

  // Allocate temporary and output buffers.
  Tensor gru_input = torch::zeros(3 * gru_state_size);
  Tensor gru_activations_hh = torch::zeros(3 * gru_state_size);
  Tensor hidden = torch::zeros(hidden_size);
  Tensor logits = torch::zeros(output_size);
  Tensor outputs = torch::zeros(num_frames * hop_length,
                                torch::TensorOptions().dtype(torch::kInt32));

  auto prev_sample_a = previous_sample.accessor<long, 1>();
  auto outputs_a = outputs.accessor<int, 1>();
  for (int timestep = 0; timestep < outputs.size(0); timestep++) {
    int frame_idx = timestep / hop_length;
    at::add_out(gru_input, sample_embeddings.index({prev_sample_a[0]}),
                gru_activations_ih.index({frame_idx}));
    at::addmv_out(gru_activations_hh, gru_bias_hh, gru_weights_hh, gru_state);
    gru_nonlinearity(gru_state, gru_input, gru_activations_hh);
    at::addmv_out(hidden, hidden_bias, hidden_weights, gru_state);
    at::relu_(hidden);
    at::addmv_out(logits, output_bias, output_weights, hidden);
    auto distribution = at::softmax(logits, 0);
    at::multinomial_out(previous_sample, distribution, 1);
    outputs_a[timestep] = prev_sample_a[0];
  }

  return outputs;
}

TORCH_LIBRARY(wavernn, m) { m.def("wavernn_inference", &wavernn_inference); }
