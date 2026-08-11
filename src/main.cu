#include "cuda_processor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

std::string GetArg(int argc, char** argv, const std::string& name,
                   const std::string& default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return default_value;
}

Image ReadPpm(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open input file: " + path.string());
    }

    std::string magic;
    int max_value = 0;
    file >> magic >> std::ws;
    if (magic != "P6") {
        throw std::runtime_error("Only binary P6 PPM is supported.");
    }

    Image image;
    file >> image.width >> image.height >> max_value;
    file.get();

    if (max_value != 255) {
        throw std::runtime_error("Only max value 255 is supported.");
    }

    image.pixels.resize(static_cast<std::size_t>(image.width) *
                        image.height * 3);
    file.read(reinterpret_cast<char*>(image.pixels.data()),
              image.pixels.size());

    if (!file) {
        throw std::runtime_error("Invalid PPM data.");
    }

    return image;
}

void WritePpm(const fs::path& path, const Image& image) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << image.width << " " << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()),
               image.pixels.size());
}

int main(int argc, char** argv) {
    try {
        const fs::path input_dir =
            GetArg(argc, argv, "--input", "data/input");
        const fs::path output_dir =
            GetArg(argc, argv, "--output", "data/output");
        const std::string mode =
            GetArg(argc, argv, "--mode", "grayscale-invert");
        const int limit =
            std::stoi(GetArg(argc, argv, "--limit", "1000"));
        const int threads =
            std::stoi(GetArg(argc, argv, "--threads", "256"));

        if (limit <= 0 || threads <= 0) {
            throw std::runtime_error("limit and threads must be positive.");
        }

        if (mode != "grayscale" && mode != "grayscale-invert") {
            throw std::runtime_error(
                "mode must be grayscale or grayscale-invert.");
        }

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable GPU was found.");
        }

        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, 0);

        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.path().extension() == ".ppm") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            throw std::runtime_error("No PPM files found in input directory.");
        }

        if (static_cast<int>(files.size()) > limit) {
            files.resize(limit);
        }

        fs::create_directories(output_dir);

        const auto wall_start = std::chrono::steady_clock::now();

        std::size_t total_pixels = 0;
        float total_kernel_ms = 0.0f;
        int processed = 0;

        for (const auto& input_path : files) {
            Image input = ReadPpm(input_path);
            Image output;
            output.width = input.width;
            output.height = input.height;
            output.pixels.resize(input.pixels.size());

            float kernel_ms = 0.0f;
            ProcessBatch(input.pixels.data(), output.pixels.data(),
                         static_cast<std::size_t>(input.width) *
                             input.height,
                         mode == "grayscale-invert", threads, &kernel_ms);

            WritePpm(output_dir / input_path.filename(), output);

            total_pixels += static_cast<std::size_t>(input.width) *
                            input.height;
            total_kernel_ms += kernel_ms;
            ++processed;
        }

        const auto wall_end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(
                wall_end - wall_start).count();

        const double images_per_second =
            processed / (total_ms / 1000.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CUDA Batch Image Processor\n";
        std::cout << "GPU: " << properties.name << "\n";
        std::cout << "Mode: " << mode << "\n";
        std::cout << "Images processed: " << processed << "\n";
        std::cout << "Total pixels: " << total_pixels << "\n";
        std::cout << "Threads per block: " << threads << "\n";
        std::cout << "Total kernel time (ms): " << total_kernel_ms << "\n";
        std::cout << "Total wall time (ms): " << total_ms << "\n";
        std::cout << "Throughput (images/sec): "
                  << images_per_second << "\n";
        std::cout << "Output directory: " << output_dir << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
