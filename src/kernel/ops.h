#pragma once

#include <torch/script.h>

namespace wavernn {

/**
 * Sample from a categorical distribution where the class probabilities are the
 * softmax of the provided logits. This is written as a single function (as
 * opposed to a softmax followed by a sampling operation) to expose optimization
 * opportunities.
 *
 * @param logits Unnormalized logits for the softmax.
 * @returns A sampled value in the range [0, n_classes - 1].
 */
int SoftmaxSampleFromLogits(const torch::Tensor &logits);

/**
 * Apply the nonlinearities for a GRU cell to update its state.
 * Besides the GRU state itself, the inputs are the pre-nonlinearity outputs of
 * the input-hidden (ih) and hidden-hidden (hh) matrix multiplies. The equations
 * for the GRU follow the PyTorch GRU as defined here:
 *
 *     https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
 *
 * The ih activations include the terms W_{ir} x_t + b_{ir}, W_{iz} x_t +
 * b_{iz}, and W_{in} x_t + b_{in}. The hh activations include the corresponding
 * terms using h_{t-1} instead of x_{t}.
 *
 * @param state The input state of shape [gru_dimension]
 * @param ih The input-hidden activations of shape [3 * gru_dimension]
 * @param hh The hidden-hidden activations of shape [3 * gru_dimension]
 */
void UpdateGruState(const torch::Tensor &state, const torch::Tensor &ih,
                    const torch::Tensor &hh);

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
              const std::vector<torch::Tensor> &sampleEmbeddings,
              const torch::Tensor &conditionerOutputs, long *sample,
              int frameIndex);

/**
 * In-place element-wise ReLU. All negative elements in the input tensor are
 * replaced with zeros.
 *
 * @param values Values to apply ReLU operation to.
 */
void ReLU(const torch::Tensor &tensor);

}  // namespace wavernn
