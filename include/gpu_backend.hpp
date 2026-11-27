#pragma once

#if defined(CUDA) && defined(HIP)
#error "CUDA and HIP cannot be enabled together"
#elif !defined(CUDA) && !defined(HIP)
#error "No GPU backend selected"
#endif

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

using GpuError = cudaError_t;
using GpuStream = cudaStream_t;
using GpuBlasHandle = cublasHandle_t;
using GpuBlasStatus = cublasStatus_t;
using GpuBlasOperation = cublasOperation_t;

#define GPU_SUCCESS cudaSuccess
#define GPU_STREAM_DEFAULT cudaStreamDefault
#define GPU_MALLOC cudaMalloc
#define GPU_FREE cudaFree
#define GPU_MEMCPY cudaMemcpy
#define GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define GPU_MEMSET cudaMemset
#define GPU_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define GPU_GET_LAST_ERROR cudaGetLastError
#define GPU_GET_ERROR_STRING cudaGetErrorString
#define GPU_LAUNCH_KERNEL(kernel, blocks, threads, shared, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), shared, stream>>>(__VA_ARGS__)

#define GPU_BLAS_SUCCESS CUBLAS_STATUS_SUCCESS
#define GPU_BLAS_OP_NONE CUBLAS_OP_N
#define GPU_BLAS_OP_TRANSPOSE CUBLAS_OP_T
#define GPU_BLAS_CREATE(handle) cublasCreate(handle)
#define GPU_BLAS_DESTROY(handle) cublasDestroy(handle)
#define GPU_BLAS_SGEMM(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    cublasSgemm(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

// Compatibility aliases let the existing HIP-style kernel launch code compile under CUDA.
#define hipError_t cudaError_t
#define hipStream_t cudaStream_t
#define hipStreamDefault cudaStreamDefault
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError
#define hipLaunchKernelGGL(kernel, blocks, threads, shared, stream, ...) \
    kernel<<<blocks, threads, shared, stream>>>(__VA_ARGS__)
#endif

#ifdef HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

using GpuError = hipError_t;
using GpuStream = hipStream_t;
using GpuBlasHandle = rocblas_handle;
using GpuBlasStatus = rocblas_status;
using GpuBlasOperation = rocblas_operation;

#define GPU_SUCCESS hipSuccess
#define GPU_STREAM_DEFAULT hipStreamDefault
#define GPU_MALLOC hipMalloc
#define GPU_FREE hipFree
#define GPU_MEMCPY hipMemcpy
#define GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define GPU_MEMSET hipMemset
#define GPU_DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define GPU_GET_LAST_ERROR hipGetLastError
#define GPU_GET_ERROR_STRING hipGetErrorString
#define GPU_LAUNCH_KERNEL(kernel, blocks, threads, shared, stream, ...) \
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), shared, stream, __VA_ARGS__)

#define GPU_BLAS_SUCCESS rocblas_status_success
#define GPU_BLAS_OP_NONE rocblas_operation_none
#define GPU_BLAS_OP_TRANSPOSE rocblas_operation_transpose
#define GPU_BLAS_CREATE(handle) rocblas_create_handle(handle)
#define GPU_BLAS_DESTROY(handle) rocblas_destroy_handle(handle)
#define GPU_BLAS_SGEMM(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    rocblas_sgemm(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#endif
