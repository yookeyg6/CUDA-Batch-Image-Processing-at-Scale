#pragma once

#include <cstddef>

void ProcessBatch(const unsigned char* host_input,
                  unsigned char* host_output,
                  std::size_t total_pixels,
                  bool invert,
                  int threads_per_block,
                  float* kernel_ms);
