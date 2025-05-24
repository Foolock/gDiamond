#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCKSIZE 1024
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void vecAddKernel_scalar(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vecAddKernel_vectorized(float *A, float *B, float *C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 tmp_A = FLOAT4(A[idx]);
        float4 tmp_B = FLOAT4(B[idx]);
        float4 tmp_C;
        tmp_C.x = tmp_A.x + tmp_B.x;
        tmp_C.y = tmp_A.y + tmp_B.y;
        tmp_C.z = tmp_A.z + tmp_B.z;
        tmp_C.w = tmp_A.w + tmp_B.w;
        FLOAT4(C[idx]) = tmp_C;
    }
}

void vecAdd_scalar(float *A, float *B, float *C, int N) {
    dim3 threadPerBlock(BLOCKSIZE);
    dim3 blockPerGrid(CEIL(N, BLOCKSIZE));
    auto start = std::chrono::steady_clock::now();
    vecAddKernel_scalar<<<blockPerGrid, threadPerBlock>>>(A, B, C, N);
    auto end = std::chrono::steady_clock::now();
    std::cout << "gpu runtime (scalar): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "s\n"; 
    cudaDeviceSynchronize();
}

void vecAdd_vectorized(float *A, float *B, float *C, int N) {
    dim3 threadPerBlock(BLOCKSIZE);
    dim3 blockPerGrid(CEIL(CEIL(N, 4), BLOCKSIZE));
    auto start = std::chrono::steady_clock::now();
    vecAddKernel_vectorized<<<blockPerGrid, threadPerBlock>>>(A, B, C, N);
    auto end = std::chrono::steady_clock::now();
    std::cout << "gpu runtime (vectorized): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "s\n"; 
    cudaDeviceSynchronize();
}

void check_result(float *C1, float *C2, int N) {
    for (int i = 0; i < N; i++) {
        if (abs(C1[i] - C2[i]) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": " << C1[i] << " vs " << C2[i] << "\n";
            return;
        }
    }
    std::cout << "Results match!\n";
}

int main() {
    int N = 1 << 20; // 1M elements

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_scalar = new float[N];
    float *h_C_vectorized = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = float(i);
        h_B[i] = float(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    vecAdd_scalar(d_A, d_B, d_C, N);
    cudaMemcpy(h_C_scalar, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    vecAdd_vectorized(d_A, d_B, d_C, N);
    cudaMemcpy(h_C_vectorized, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_result(h_C_scalar, h_C_vectorized, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_scalar;
    delete[] h_C_vectorized;

    return 0;
}

