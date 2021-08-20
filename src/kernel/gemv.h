#pragma once

#include <torch/script.h>

#include <vector>

namespace wavernn {

/**
 * Our WaveRNN inference kernel can be accelerated through the use of
 * block-sparse weight matrices. A block-sparse weight matris is a matrix where
 * rectangular blocks of values are all either zero or non-zero. If a large
 * fraction (>50%) of the weights in a matrix are zero, we can efficiently
 * multiply by that matrix by skipping the operations that would multiply by
 * zero.
 *
 * In order to efficiently implement block-sparse matrix-vector multiplies, we
 * need to reorganize the matrix data in memory. Without this, the memory access
 * is effectively random, leading to poor performance due to cache misses.
 *
 * This class represents a weight matrix (with a bias) which has been repacked
 * in memory to provide for highly efficient matrix-vector multiplication. After
 * the matrix has been created, the gemv() function (gemv for GEneral
 * Matrix-Vector multiply) may be used to multiply a vector by the given matrix.
 */
class PackedLinear {
 public:
  /// Create a new packed packed matrix. The sparsity of the matrix is detected
  /// automatically, so for maximum efficiency, the matrix should be a
  /// block-sparse matrix with sparsity >= 50%.
  /// @param matrix The float32 matrix of shape [output_size, input_size].
  /// @param bias The float32 bias vector of shape [output_size].
  /// @param quantize If true, use quantized kernels automatically when
  /// possible.
  PackedLinear(const torch::Tensor& matrix, const torch::Tensor& bias,
               bool quantized = true);

  /// Multiply a vector by this matrix.
  /// @param out Output float32 tensor of shape [output_size].
  /// @param vector Input float32 vector of shape [input_size].
  void gemv(torch::Tensor& out, const torch::Tensor& vector) const;

 private:
  /// Whether quantized kernels should be used when possible.
  bool quantized_;

  /// The output size of the matrix.
  int output_size_;

  /// The input size of the matrix.
  int input_size_;

  // Data for float32 multiplication.

  /// The packed float32 weight data.
  std::vector<float> data_;

  /// The number of non-zero input blocks for each output block.
  std::vector<int> blocksPerRow_;

  /// The offset in the weight data for each row's blocks.
  /// This is stored explicitly so that multithreaded gemv does not need to
  /// compute where to start reading data for each thread.
  std::vector<int> rowOffsets_;

  /// The input vector indices which each block should be multiplied by.
  std::vector<int> indices_;

  /// The original matrix of size [output_size, input_size].
  /// Used if we can't use our AVX sparse gemv.
  const torch::Tensor& matrix_;

  /// The bias tensor of size [output_size].
  const torch::Tensor& bias_;

  // Data for int16 multiplication.

  /// The packed int16 weight data.
  std::vector<int16_t> quantizedData_;

  /// The number of non-zero input blocks for each output block.
  std::vector<int> quantizedBlocksPerRow_;

  /// The offset in the weight data for each row's blocks.
  /// This is stored explicitly so that multithreaded gemv does not need to
  /// compute where to start reading data for each thread.
  std::vector<int> quantizedRowOffsets_;

  /// The input vector indices which each block should be multiplied by.
  std::vector<int> quantizedIndices_;

  /// The float scales for each row.
  std::vector<float> quantizedRowScales_;
};

}  // namespace wavernn
