#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <string>

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(1);
    }
}

void checkCublas(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %s\n", msg);
        exit(1);
    }
}


// Matrix size will be set based on mode
int get_matrix_size(const char* mode) {
    if (strcmp(mode, "high") == 0) {
        return 8192;
    } else if (strcmp(mode, "medium") == 0) {
        return 4096;
    } else if (strcmp(mode, "low") == 0) {
        return 1024;
    } else {
        fprintf(stderr, "Error: Invalid mode in get_matrix_size: '%s'.\n", mode);
        exit(1);
    }
}

void run_stress(const char* mode) {
    int N = get_matrix_size(mode);
    float *A, *B, *C;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Allocate matrices on GPU
    checkCuda(cudaMalloc((void**)&A, N*N*sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc((void**)&B, N*N*sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc((void**)&C, N*N*sizeof(float)), "cudaMalloc C");

    float alpha = 1.0f, beta = 0.0f;

    printf("Starting GPU stress benchmark: infinite matrix multiplication of %dx%d, mode=%s\n", N, N, mode);
    int i = 0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, A, N, B, N, &beta, C, N), "cublasSgemm");
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (i % 10 == 0) {
            printf("Iteration %d: %.2f ms\n", i, ms);
        }
        // Utilization control
        if (strcmp(mode, "high") == 0) {
            // No sleep, maximize utilization
        } else if (strcmp(mode, "medium") == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(2))); // sleep half compute time
        } else if (strcmp(mode, "low") == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(10))); // sleep double compute time
        } else {
            fprintf(stderr, "Error: Invalid mode in run_stress: '%s'.\n", mode);
            exit(1);
        }
        ++i;
    }
    // Never reached, but cleanup for completeness
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    // Usage: gpu_stress_benchmark [utilization]
    // Accepts: low, medium, high
    if (argc < 2) {
        fprintf(stderr, "Error: Utilization argument required.\n");
        fprintf(stderr, "Usage: %s [low|medium|high]\n", argv[0]);
        return 1;
    }
    const char* mode = NULL;
    if (strcmp(argv[1], "low") == 0) {
        mode = "low";
    } else if (strcmp(argv[1], "medium") == 0) {
        mode = "medium";
    } else if (strcmp(argv[1], "high") == 0) {
        mode = "high";
    } else {
        fprintf(stderr, "Error: Invalid utilization argument '%s'.\n", argv[1]);
        fprintf(stderr, "Usage: %s [low|medium|high]\n", argv[0]);
        return 1;
    }
    run_stress(mode);
    return 0;
}