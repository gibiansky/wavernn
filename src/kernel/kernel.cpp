#include "kernel.h"

#include <immintrin.h>
#include <mkl/mkl.h>
#include <omp.h>
#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using torch::Tensor;

__m256 _mm256_tanh_ps(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 signmask = _mm256_set1_ps(-0.0f);
  const __m256 max_val = _mm256_set1_ps(30.0f);

  const __m256 c1 = _mm256_set1_ps(6.931471825e-01);
  const __m256 c2 = _mm256_set1_ps(2.402264923e-01);
  const __m256 c3 = _mm256_set1_ps(5.550357327e-02);
  const __m256 c4 = _mm256_set1_ps(9.618237615e-03);
  const __m256 c5 = _mm256_set1_ps(1.339077600e-03);
  const __m256 c6 = _mm256_set1_ps(1.540359954e-04);

  __m256 signs = _mm256_and_ps(x, signmask);
  x = _mm256_andnot_ps(signmask, x);
  x = _mm256_min_ps(x, max_val);

  __m256 f = x;

  __m256i i = _mm256_cvtps_epi32(f);
  f = _mm256_sub_ps(f, _mm256_cvtepi32_ps(i));

  __m256 p = c6;
  p = _mm256_fmadd_ps(p, f, c5);
  p = _mm256_fmadd_ps(p, f, c4);
  p = _mm256_fmadd_ps(p, f, c3);
  p = _mm256_fmadd_ps(p, f, c2);
  p = _mm256_fmadd_ps(p, f, c1);
  i = _mm256_slli_epi32(i, 23);

  __m256 biased_expm =
      _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_castps_si256(one), i));
  __m256 exp_cor = _mm256_sub_ps(one, biased_expm);
  __m256 exp_cor_p = _mm256_add_ps(one, biased_expm);

  __m256 exp2xm1 = _mm256_xor_ps(signs, _mm256_fmadd_ps(p, f, exp_cor));
  __m256 exp2xp1 = _mm256_fmadd_ps(p, f, exp_cor_p);
  return _mm256_div_ps(exp2xm1, exp2xp1);
}

__m256 _mm256_exp_ps(__m256 _x) {
  __m256 c1 = _mm256_set1_ps(0.007972914726F);
  __m256 c2 = _mm256_set1_ps(0.1385283768F);
  __m256 c3 = _mm256_set1_ps(2.885390043F);
  __m256 c4 = _mm256_set1_ps(1.442695022F);
  __m256 x = _mm256_mul_ps(_x, c4);  // convert to 2^(x)
  __m256 intPartf = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  x = _mm256_sub_ps(x, intPartf);
  __m256 xx = _mm256_mul_ps(x, x);
  __m256 a = _mm256_add_ps(
      x, _mm256_mul_ps(c1, _mm256_mul_ps(xx, x)));  // can be improved with FMA
  __m256 b = _mm256_add_ps(c3, _mm256_mul_ps(c2, xx));
  __m256 res = _mm256_div_ps(_mm256_add_ps(b, a), _mm256_sub_ps(b, a));
  __m256i intPart = _mm256_cvtps_epi32(
      intPartf);  // res = 2^intPart. Can be improved with AVX2!
  __m128i ii0 = _mm_slli_epi32(_mm256_castsi256_si128(intPart), 23);
  __m128i ii1 = _mm_slli_epi32(_mm256_extractf128_si256(intPart, 1), 23);
  __m128i res_0 =
      _mm_add_epi32(ii0, _mm256_castsi256_si128(_mm256_castps_si256(res)));
  __m128i res_1 =
      _mm_add_epi32(ii1, _mm256_extractf128_si256(_mm256_castps_si256(res), 1));
  return _mm256_insertf128_ps(
      _mm256_castsi256_ps(_mm256_castsi128_si256(res_0)),
      _mm_castsi128_ps(res_1), 1);
}

/**
 * A timer utility, used for recording how long C++ inference kernels spend in
 * different parts of the kernel.  Timers allow you to separate your kernel
 * into sections, labeled with human-readable string names, and record the
 * duration of the sections as the kernel is running.  If a section is entered
 * multiple times, the times are aggregated.
 *
 * For example:
 *
 *     // Create the timer.
 *     bool enableTimer = true;
 *     Timer timer(enableTimer);
 *
 *     // Use the timer in a few sections.
 *     timer.start("Section 1");
 *     doSomething();
 *
 *     for(int i = 0; i < 1000; i++) {
 *         timer.start("Second Section");
 *         doSomethingElse();
 *
 *         timer.start("Sec 3");
 *         something();
 *     }
 *
 *     // Print a breakdown of times.
 *     timer.print();
 *
 * This example will print a table along the lines of:
 *
 *     WaveRNN Kernel Timings
 *     ======================
 *     Section 1: 3 ms (0%)
 *     Second Section: 193 ms (14%)
 *     Sec 3: 1179 ms (86%)
 *     ======================
 *     Total: 1372 ms
 *     ======================
 */
class Timer {
 public:
  /// Create a new timer.
  /// @param timing whether to enable this timer. If a timer is not enabled,
  /// its methods do nothing. Turn off timers to reduce overhead.
  explicit Timer(bool timing) : enabled(timing) {}

  /// Enter a timer section. This resets the timer duration back to zero and
  /// starts timing a new section. If this section has been encountered before,
  /// the duration is accumulated.
  ///
  /// @param key Name of the section. A char* is used instead of an std::string
  /// to avoid overhead from memory allocation, copying, and comparison. This
  /// method is intended to be used with string literals.
  void start(const char *key) {
    // If the timer is disabled, every method should do nothing.
    if (!enabled) {
      return;
    }

    // Stop timing the previous section.
    stop();

    // Start timing a new section.
    currentSectionName = key;
    startTime = std::chrono::steady_clock::now();
  }

  /// Stop timing the current section. Starting a new section or printing output
  /// does this implicitly, so this is only necessary if you intend to do
  /// something you explicitly do not want to include in your timings.
  void stop() {
    // If the timer is disabled, every method should do nothing.
    if (!enabled) {
      return;
    }

    // If no section is active, we can't stop timing.
    if (currentSectionName == nullptr) {
      return;
    }

    // Get the elapsed time in milliseconds since the last start() call.
    auto elapsedTime = std::chrono::steady_clock::now() - startTime;
    auto elapsedNanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsedTime);
    float elapsedMs = elapsedNanos.count() / 1e6f;

    // If this is the first time we encounter this section, set it's time,
    // otherwise increment the existing time.
    if (timings.find(currentSectionName) == timings.end()) {
      timings[currentSectionName] = elapsedMs;
      keys.push_back(currentSectionName);
    } else {
      timings[currentSectionName] += elapsedMs;
    }
  }

  /// Log a display of time elapsed in each section to stdout.
  /// The display will look roughly like this:
  ///
  ///    WaveRNN Kernel Timings
  ///    ======================
  ///    Section 1: 3 ms (0%)
  ///    Second Section: 193 ms (14%)
  ///    Sec 3: 1179 ms (86%)
  ///    ======================
  ///    Total: 1372 ms
  ///    ======================
  void print() {
    // If the timer is disabled, every method should do nothing.
    if (!enabled) {
      return;
    }

    // Stop timing whatever section is active.
    stop();

    // Compute the total to be able to compute percentages.
    float total = 0.0f;
    for (const char *key : keys) {
      total += timings[key];
    }

    // Display the table with milliseconds and percentages.
    std::cout << std::setprecision(0) << std::fixed;
    std::cout << "WaveRNN Kernel Timings" << std::endl;
    std::cout << "======================" << std::endl;
    for (const char *key : keys) {
      std::cout << key << ": " << timings[key] << " ms ("
                << timings[key] / total * 100 << "%)" << std::endl;
    }
    std::cout << "======================" << std::endl;
    std::cout << "Total: " << total << " ms" << std::endl;
    std::cout << "======================" << std::endl;
  }

 private:
  /// Whether or not to enable this timer. When not enabled, all methods do
  /// nothing. This flag allows users to easily minimize the overhead incurred
  /// by the timer.
  bool enabled;

  /// The currently active section name. When no section is active, this is
  /// nullptr.
  ///
  /// This uses a char* instead of an std::string to minimize overhead.
  //// Working with std::string incurs overhead due to copying, memory
  //// allocation, and string comparison, while working with a char* is less
  //// safe but much more efficient.
  const char *currentSectionName = nullptr;

  /// The time at which the last section began.
  std::chrono::steady_clock::time_point startTime =
      std::chrono::steady_clock::now();

  /// Aggregated timings across the sections. Keys are string section names
  /// and values are times measured in milliseconds.
  std::unordered_map<const char *, float> timings;

  /// The list of keys that have been used. These are stored in a separate
  /// vector so that when the timings are printed out, they can be printed in
  /// the same order as the sections were encountered.
  std::vector<const char *> keys;
};

int SampleFromSoftmax(const Tensor &distribution) {
  static std::vector<float> random;
  static size_t idx = 0;
  if (random.empty()) {
    random.resize(10000);

    std::default_random_engine generator;  // NOLINT
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (float &flt : random) {
      flt = distribution(generator);
    }
  }

  float rand = random[idx];
  idx = (idx + 1) % random.size();

  float sum = 0.0;
  const float *dist = distribution.data_ptr<float>();
  int dim = distribution.size(0);
  for (int i = 0; i < dim; i++) {
    sum += dist[i];
    if (sum >= rand) {
      return i;
    }
  }
  return dim - 1;
}

__m256 gru_nonlinearity(__m256 state, __m256 r_ih, __m256 r_hh, __m256 z_ih,
                        __m256 z_hh, __m256 n_ih, __m256 n_hh) {
  // Sum input and recurrent pre-activations.
  auto r_act = _mm256_add_ps(r_ih, r_hh);
  auto z_act = _mm256_add_ps(z_ih, z_hh);

  // Negate activations.
  auto zero = _mm256_set1_ps(0.0);
  r_act = _mm256_add_ps(zero, r_act);
  z_act = _mm256_add_ps(zero, z_act);

  // Take e^-x.
  r_act = _mm256_exp_ps(r_act);
  z_act = _mm256_exp_ps(z_act);

  // Add one.
  auto one = _mm256_set1_ps(1.0);
  r_act = _mm256_add_ps(one, r_act);
  z_act = _mm256_add_ps(one, z_act);

  // Final sigmoid.
  auto r = _mm256_rcp_ps(r_act);
  auto z = _mm256_rcp_ps(z_act);

  // Compute n activation.
  auto n_act = _mm256_mul_ps(r, n_hh);
  n_act = _mm256_add_ps(n_act, n_ih);
  auto n = _mm256_tanh_ps(n_act);

  auto one_minus_z = _mm256_sub_ps(one, z);
  return _mm256_add_ps(_mm256_mul_ps(one_minus_z, n), _mm256_mul_ps(z, state));
}

void gru_nonlinearity(const Tensor &gru_state, const Tensor &gru_activations_ih,
                      const Tensor &gru_activations_hh) {
  const int size = gru_state.size(0);
  float *ih = gru_activations_ih.data_ptr<float>();
  float *hh = gru_activations_hh.data_ptr<float>();
  float *state = gru_state.data_ptr<float>();

  if (true) {  // NOLINT
    ASSERT_BOOL(size % 8 == 0);
#pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
      auto r_ih = _mm256_loadu_ps(ih + i);
      auto z_ih = _mm256_loadu_ps(ih + i + size);
      auto n_ih = _mm256_loadu_ps(ih + i + 2 * size);
      auto r_hh = _mm256_loadu_ps(hh + i);
      auto z_hh = _mm256_loadu_ps(hh + i + size);
      auto n_hh = _mm256_loadu_ps(hh + i + 2 * size);
      auto state_vec = _mm256_loadu_ps(state + i);
      state_vec =
          gru_nonlinearity(state_vec, r_ih, r_hh, z_ih, z_hh, n_ih, n_hh);
      _mm256_storeu_ps(state + i, state_vec);
    }
    return;
  }

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

inline void floatRelu(const Tensor &tensor) {
  float *h = tensor.data_ptr<float>();
  int size = tensor.size(0);
  for (int i = 0; i < size; i++) {
    h[i] = h[i] < 0 ? 0.0f : h[i];
  }
}

#define USE_SPARSE_GEMV

Tensor wavernn_inference(
    /* WaveRNN inputs */
    const Tensor &gru_activations_ih, const Tensor &previous_sample,
    const Tensor &gru_state,

    /* WaveRNN weights */
    const Tensor &sample_embeddings, const Tensor &gru_weights_hh,
    const Tensor &gru_bias_hh, const Tensor &hidden_weights,
    const Tensor &hidden_bias, const Tensor &output_weights,
    const Tensor &output_bias,

    /* WaveRNN hyperparameters */
    const int64_t hop_length, const bool timing) {
  Timer timer(timing);
  timer.start("Initialization");
  c10::InferenceMode inferenceMode;

  // Extract hyperparameters.
  auto num_frames = gru_activations_ih.size(0);
  auto gru_state_size = gru_state.size(0);
  auto hidden_size = hidden_bias.size(0);
  auto output_size = output_bias.size(0);

  // Verify sizes.
  ASSERT_TENSOR_SIZE(gru_activations_ih, num_frames, 3 * gru_state_size);
  ASSERT_TENSOR_SIZE(previous_sample, 1);
  ASSERT_TENSOR_SIZE(gru_state, gru_state_size);
  ASSERT_TENSOR_SIZE(sample_embeddings, output_size, 3 * gru_state_size);
  ASSERT_TENSOR_SIZE(gru_weights_hh, 3 * gru_state_size, gru_state_size);
  ASSERT_TENSOR_SIZE(gru_bias_hh, 3 * gru_state_size);
  ASSERT_TENSOR_SIZE(hidden_weights, hidden_size, gru_state_size);
  ASSERT_TENSOR_SIZE(hidden_bias, hidden_size);
  ASSERT_TENSOR_SIZE(output_weights, output_size, hidden_size);
  ASSERT_TENSOR_SIZE(output_bias, output_size);

  // Allocate temporary and output buffers.
  Tensor gru_input = torch::zeros(3 * gru_state_size);
  Tensor gru_activations_hh = torch::zeros(3 * gru_state_size);
  Tensor hidden = torch::zeros(hidden_size);
  Tensor logits = torch::zeros(output_size);
  Tensor outputs = torch::zeros(num_frames * hop_length,
                                torch::TensorOptions().dtype(torch::kInt32));

  auto prev_sample_a = previous_sample.accessor<long, 1>();
  auto outputs_a = outputs.accessor<int, 1>();

  float *sample_embeddings_ptr = sample_embeddings.data_ptr<float>();
  float *gru_activations_ih_ptr = gru_activations_ih.data_ptr<float>();
  float *gru_input_ptr = gru_input.data_ptr<float>();
  int gru_input_size = gru_input.size(0);

#ifdef USE_SPARSE_GEMV
  SparsePackedMatrix gruMat(gru_weights_hh, gru_bias_hh);
  SparsePackedMatrix hiddenMat(hidden_weights, hidden_bias);
  SparsePackedMatrix outputMat(output_weights, output_bias);
#endif

  for (int timestep = 0; timestep < outputs.size(0); timestep++) {
    int frame_idx = timestep / int(hop_length);
    timer.start("GRU Input");
    vsAdd(gru_input_size,
          sample_embeddings_ptr + prev_sample_a[0] * gru_input_size,
          gru_activations_ih_ptr + frame_idx * gru_input_size, gru_input_ptr);

    timer.start("GRU GEMV");
#ifdef USE_SPARSE_GEMV
    gruMat.gemv(gru_activations_hh, gru_state);
#else
    at::addmv_out(gru_activations_hh, gru_bias_hh, gru_weights_hh, gru_state);
#endif

    timer.start("GRU Nonlinearities");
    gru_nonlinearity(gru_state, gru_input, gru_activations_hh);

    timer.start("Hidden GEMV");
#ifdef USE_SPARSE_GEMV
    hiddenMat.gemv(hidden, gru_state);
#else
    at::addmv_out(hidden, hidden_bias, hidden_weights, gru_state);
#endif

    timer.start("Hidden ReLU");
    floatRelu(hidden);

    timer.start("Output GEMV");
#ifdef USE_SPARSE_GEMV
    outputMat.gemv(logits, hidden);
#else
    at::addmv_out(logits, output_bias, output_weights, hidden);
#endif

    timer.start("Softmax");
    auto distribution = at::softmax(logits, 0);

    timer.start("Sample");
    auto sample = SampleFromSoftmax(distribution);
    prev_sample_a[0] = sample;
    outputs_a[timestep] = sample;
  }

  timer.print();

  return outputs;
}

Tensor sparse_gemv(const Tensor &bias, const Tensor &matrix,
                   const Tensor &vector) {
  c10::InferenceMode inferenceMode;
  SparsePackedMatrix mat(matrix, bias);
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
  SparsePackedMatrix mat(matrix, bias);

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

TORCH_LIBRARY(wavernn, m) {  // NOLINT
  m.def("wavernn_inference", &wavernn_inference);
  m.def("sparse_gemv", &sparse_gemv);
  m.def("sparse_gemv_benchmark", &sparse_gemv_benchmark);
}
