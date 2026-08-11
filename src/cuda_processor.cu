#include "cuda_processor.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

__global__ void GrayscaleKernel(const unsigned char* input,
                                unsigned char* output,
                                std::size_t pixels,
                                bool invert) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= pixels) {
        return;
    }

    const std::size_t rgb = index * 3;
    const float gray = 0.299f * input[rgb] +
                       0.587f * input[rgb + 1] +
                       0.114f * input[rgb + 2];

    unsigned char value = static_cast<unsigned char>(gray);
    if (invert) {
        value = static_cast<unsigned char>(255 - value);
    }

    output[rgb] = value;
    output[rgb + 1] = value;
    output[rgb + 2] = value;
}

void CheckCuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void ProcessBatch(const unsigned char* host_input,
                  unsigned char* host_output,
                  std::size_t total_pixels,
                  bool invert,
                  int threads_per_block,
                  float* kernel_ms) {
    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;
    const std::size_t bytes = total_pixels * 3;

    CheckCuda(cudaMalloc(&device_input, bytes), "cudaMalloc input");
    CheckCuda(cudaMalloc(&device_output, bytes), "cudaMalloc output");

    CheckCuda(cudaMemcpy(device_input, host_input, bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy input");

    const int blocks = static_cast<int>(
        (total_pixels + threads_per_block - 1) / threads_per_block);

    cudaEvent_t start;
    cudaEvent_t stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate start");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    CheckCuda(cudaEventRecord(start), "cudaEventRecord start");

    GrayscaleKernel<<<blocks, threads_per_block>>>(
        device_input, device_output, total_pixels, invert);

    CheckCuda(cudaGetLastError(), "kernel launch");
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    CheckCuda(cudaEventElapsedTime(kernel_ms, start, stop),
              "cudaEventElapsedTime");

    CheckCuda(cudaMemcpy(host_output, device_output, bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy output");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_input);
    cudaFree(device_output);
}
