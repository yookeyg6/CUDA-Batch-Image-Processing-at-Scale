CUDA_ARCH ?=
NVCC ?= nvcc
CXXFLAGS = -O2 -std=c++17
TARGET = cuda_batch_processor

all: $(TARGET)

$(TARGET): src/main.cu src/cuda_processor.cu
	$(NVCC) $(CXXFLAGS) -Iinclude $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
