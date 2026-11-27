# Compilatore HIP per AMD ROCm (C++ / GPU)
HIPCC ?= hipcc

BACKEND ?= HIP

# Directory base
SRC_DIR := src
CKPT_DIR := checkpoints

ifeq ($(BACKEND),HIP)
COMPILER := $(HIPCC)
GPUFLAGS := -DHIP -I/opt/rocm/include
GPULIBS := -L/opt/rocm/lib -lrocblas
BUILD_DIR := build/hip
else ifeq ($(BACKEND),CUDA)
COMPILER := nvcc
GPUFLAGS := -DCUDA $(if $(NVCC_CCBIN),-ccbin "$(NVCC_CCBIN)")
GPULIBS := -lcublas
BUILD_DIR := build/cuda
else
$(error BACKEND must be HIP or CUDA)
endif

# Flag diversi per GCC/Clang (hipcc) vs NVCC (nvcc richiede -Xcompiler per flag GCC)
ifeq ($(BACKEND),CUDA)
CXXFLAGS := -x cu -std=c++17 -O3 -Iinclude $(GPUFLAGS) -Xcompiler "-Wall -Wextra"
else
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude $(GPUFLAGS)
endif

# Sorgenti e file oggetto
SRCS_CPP := $(SRC_DIR)/dataset.cpp \
            $(SRC_DIR)/tensor.cpp \
            $(SRC_DIR)/config.cpp \
            $(SRC_DIR)/model.cpp \
            $(SRC_DIR)/train.cpp \
            $(SRC_DIR)/bpe_tokenizer.cpp

OBJS_CPP := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS := $(OBJS_CPP)

TARGET := $(BUILD_DIR)/train
PREPARE_BPE_SRCS := $(SRC_DIR)/bpe_tokenizer.cpp $(SRC_DIR)/prepare_bpe.cpp
PREPARE_BPE_BIN := $(BUILD_DIR)/prepare_bpe
GENERATE_SRCS := $(SRC_DIR)/generate.cpp $(SRC_DIR)/model.cpp $(SRC_DIR)/bpe_tokenizer.cpp $(SRC_DIR)/tensor.cpp $(SRC_DIR)/config.cpp
GENERATE_BIN := $(BUILD_DIR)/generate

.PHONY: all dirs clean clean-all run
all: dirs $(TARGET) $(PREPARE_BPE_BIN) $(GENERATE_BIN)

dirs:
	@mkdir -p $(BUILD_DIR) $(CKPT_DIR) data/raw

$(PREPARE_BPE_BIN): $(PREPARE_BPE_SRCS) include/bpe_tokenizer.hpp
	$(CXX) -std=c++17 -O3 -Wall -Wextra -Iinclude $(PREPARE_BPE_SRCS) -o $@

$(GENERATE_BIN): $(GENERATE_SRCS) include/model.hpp include/kernels.hpp include/tensor.hpp include/gpu_backend.hpp
	$(COMPILER) $(CXXFLAGS) $(GENERATE_SRCS) $(GPULIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp include/gpu_backend.hpp
	$(COMPILER) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(COMPILER) $(OBJS) $(GPULIBS) -o $@

run: all
ifeq ($(BACKEND),HIP)
	HSA_OVERRIDE_GFX_VERSION=10.3.0 $(TARGET)
else
	$(TARGET)
endif

clean:
	rm -rf $(BUILD_DIR)

clean-all:
	rm -rf build
