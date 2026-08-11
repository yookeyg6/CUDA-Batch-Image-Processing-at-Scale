# CUDA Batch Image Processor

A compact CUDA project that processes a large batch of small RGB images on the GPU. Each image is converted to grayscale and then inverted by a custom CUDA kernel.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (`nvcc`)
- CMake 3.18+

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

## Run

From the project root:

```bash
./build/cuda_batch_processor --input data/input --output data/output --mode grayscale-invert
```

Optional arguments:

```text
--input DIR        Input PPM directory
--output DIR       Output PPM directory
--mode MODE        grayscale | grayscale-invert
--limit N          Maximum number of images to process
--threads N        CUDA threads per block (default: 256)
```

Example:

```bash
./build/cuda_batch_processor --input data/input --output data/output --mode grayscale-invert --limit 1000 --threads 256
```

## Dataset

The repository contains a small PPM dataset generator so the project can be reproduced without external downloads. The generator creates hundreds of deterministic RGB test images.

Generate 1000 images:

```bash
python3 scripts/generate_dataset.py --output data/input --count 1000 --width 128 --height 128
```

## GPU computation

The main processing is performed by a custom CUDA `__global__` kernel. Each CUDA thread processes one pixel, computes grayscale intensity, and optionally inverts the resulting value. The kernel is launched with a configurable block size.

CUDA kernels are compiled with `nvcc`.

## Output and evidence

The program prints:

- CUDA device name
- number of images
- image dimensions
- number of processed pixels
- GPU kernel time
- total execution time
- throughput

Save a reproducible execution log:

```bash
mkdir -p artifacts
./build/cuda_batch_processor --input data/input --output data/output --mode grayscale-invert --limit 1000 --threads 256 | tee artifacts/execution_log.txt
```

Copy a few input/output images into `artifacts/` for visual proof.

## Project structure

```text
cuda_batch_image_project/
├── README.md
├── CMakeLists.txt
├── Makefile
├── run.sh
├── include/
│   └── cuda_processor.h
├── src/
│   ├── main.cu
│   └── cuda_processor.cu
├── scripts/
│   └── generate_dataset.py
├── data/
│   ├── input/
│   └── output/
└── artifacts/
```

## Notes

This project intentionally uses the simple PPM image format so that the focus remains on CUDA processing rather than third-party image libraries. The GPU performs the pixel transformation; file I/O remains on the host.
