CXX ?= g++
NVCC ?= nvcc

EXE_EXT :=
ifeq ($(OS),Windows_NT)
EXE_EXT := .exe
MKDIR_P = if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
RM_RF = if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"
else
MKDIR_P = mkdir -p $(BUILD_DIR)
RM_RF = rm -rf $(BUILD_DIR)
endif

BUILD_DIR := build
CPU_TARGET := $(BUILD_DIR)/rd_cpu$(EXE_EXT)
GPU_TARGET := $(BUILD_DIR)/rd_cuda$(EXE_EXT)
COMMON_SOURCES := src/main.cpp src/common.cpp src/cpu_backend.cpp
CUDA_SOURCES := $(COMMON_SOURCES) src/gpu_backend.cu

CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -std=c++17 -O3 -DRD_ENABLE_CUDA

.PHONY: all gpu cpu run run-cpu clean

all: gpu

$(BUILD_DIR):
	$(MKDIR_P)

gpu: $(GPU_TARGET)

$(GPU_TARGET): $(CUDA_SOURCES) src/reaction_diffusion.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SOURCES) -o $(GPU_TARGET)

cpu: $(CPU_TARGET)

$(CPU_TARGET): $(COMMON_SOURCES) src/reaction_diffusion.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_SOURCES) -o $(CPU_TARGET)

run: gpu
	$(GPU_TARGET) --backend gpu --width 1024 --height 1024 \
		--steps 5000 --output-interval 500 --preset coral \
		--output-dir proof/gpu_demo

run-cpu: cpu
	$(CPU_TARGET) --backend cpu --width 256 --height 256 \
		--steps 800 --output-interval 200 --preset coral \
		--output-dir proof/cpu_reference

clean:
	$(RM_RF)
