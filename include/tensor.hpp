#pragma once

#include "gpu_backend.hpp"
#include <cstddef>
#include <vector>

class GpuTensor {
public:
    GpuTensor() = default;

    // Alloca `num_elements` float sulla VRAM della GPU (non inizializzati)
    explicit GpuTensor(std::size_t num_elements);

    // Libera automaticamente la memoria GPU quando l'oggetto esce di scope (RAII)
    ~GpuTensor();

    // Disabilita la copia (evita doppie hipFree sullo stesso puntatore GPU)
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    // Abilita lo spostamento (move): trasferisce la proprietà del puntatore
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    // Rialloca o ridimensiona solo se cambia il numero di elementi
    void resize(std::size_t num_elements);

    // Azzera tutti i byte in VRAM (utile per i gradienti a ogni step)
    void zero();

    // Copia dati dalla RAM (Host) alla VRAM (Device)
    void copy_from_host(const float* src, std::size_t count);
    void copy_from_host(const std::vector<float>& src);

    // Copia dati dalla VRAM (Device) alla RAM (Host)
    void copy_to_host(float* dst, std::size_t count) const;
    std::vector<float> to_host() const;

    float* data() { return data_; }
    const float* data() const { return data_; }
    std::size_t size() const { return size_; }
    std::size_t bytes() const { return size_ * sizeof(float); }

private:
    float* data_ = nullptr;
    std::size_t size_ = 0;

    void release();
};