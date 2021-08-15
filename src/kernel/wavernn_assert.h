#pragma once

// Assert that a tensor's size matches what we expect.
#define STRINGIFY(X) #X
#define ASSERT_TENSOR_SIZE(_tensor, ...)                                       \
  {                                                                            \
    if ((_tensor).sizes() != c10::IntArrayRef{__VA_ARGS__}) {                  \
      throw std::runtime_error(__FILE__ ":" STRINGIFY(                         \
          __LINE__) " Tensor " #_tensor " size does not match " #__VA_ARGS__); \
    }                                                                          \
  }

#define ASSERT_BOOL(_cond)                                            \
  {                                                                   \
    if (!(_cond)) {                                                   \
      throw std::runtime_error(                                       \
          __FILE__ ":" STRINGIFY(__LINE__) " Failed check: " #_cond); \
    }                                                                 \
  }
