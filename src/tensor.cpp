#include "tensor.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
inline void check_gpu(GpuError err, const char* file, int line) {
    if (err != GPU_SUCCESS) {
        std::fprintf(stderr, "GPU Error [%s:%d]: %s\n",
                     file, line, GPU_GET_ERROR_STRING(err));
        std::exit(EXIT_FAILURE);
    }
}
} // namespace

#define GPU_ASSERT(cmd) check_gpu((cmd), __FILE__, __LINE__)

GpuTensor::GpuTensor(std::size_t num_elements) {
    resize(num_elements);
}

GpuTensor::~GpuTensor() {
    release();
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void GpuTensor::release() {
    if (data_ != nullptr) {
        GpuError err = GPU_FREE(data_);
        if (err != GPU_SUCCESS) {
            std::fprintf(stderr, "Attenzione: GPU free fallita: %s\n",
                         GPU_GET_ERROR_STRING(err));
        }
        data_ = nullptr;
        size_ = 0;
    }
}

void GpuTensor::resize(std::size_t num_elements) {
    if (num_elements == size_) {
        return;
    }

    release();
    if (num_elements == 0) {
        return;
    }

    GPU_ASSERT(GPU_MALLOC(reinterpret_cast<void**>(&data_), num_elements * sizeof(float)));
    size_ = num_elements;
}

void GpuTensor::zero() {
    if (data_ != nullptr && size_ > 0) {
        GPU_ASSERT(GPU_MEMSET(data_, 0, bytes()));
    }
}

void GpuTensor::copy_from_host(const float* src, std::size_t count) {
    if (count == 0) return;
    if (count > size_) {
        std::fprintf(stderr, "Errore: copy_from_host count (%zu) > size_ (%zu)\n",
                     count, size_);
        std::exit(EXIT_FAILURE);
    }
    GPU_ASSERT(GPU_MEMCPY(data_, src, count * sizeof(float), GPU_MEMCPY_HOST_TO_DEVICE));
}

void GpuTensor::copy_from_host(const std::vector<float>& src) {
    copy_from_host(src.data(), src.size());
}

void GpuTensor::copy_to_host(float* dst, std::size_t count) const {
    if (count == 0) return;
    if (count > size_) {
        std::fprintf(stderr, "Errore: copy_to_host count (%zu) > size_ (%zu)\n",
                     count, size_);
        std::exit(EXIT_FAILURE);
    }
    GPU_ASSERT(GPU_MEMCPY(dst, data_, count * sizeof(float), GPU_MEMCPY_DEVICE_TO_HOST));
}

std::vector<float> GpuTensor::to_host() const {
    std::vector<float> out(size_);
    copy_to_host(out.data(), size_);
    return out;
}