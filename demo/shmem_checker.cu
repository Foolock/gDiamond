#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void max_shmem_kernel(float* out, int N) {
    extern __shared__ float shmem[];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        shmem[tid] = tid * 2.0f;
    }
    __syncthreads();

    if (tid < N) {
        out[tid] = shmem[tid];
    }
}

void cpu_version(std::vector<float>& ref, int N) {
    for (int i = 0; i < N; ++i) {
        ref[i] = i * 2.0f;
    }
}

int main() {
    const int N = 24816; // 99264 bytes / sizeof(float)
    const int threadsPerBlock = 256;
    const size_t shmemBytes = N * sizeof(float);

    std::vector<float> host_ref(N);
    std::vector<float> host_out(N);

    float* d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Allow large shared memory size
    cudaFuncSetAttribute(
        max_shmem_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmemBytes
    );

    size_t grid_size = (N + threadsPerBlock - 1) / threadsPerBlock;
    max_shmem_kernel<<<grid_size, threadsPerBlock, shmemBytes>>>(d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU computation
    cpu_version(host_ref, N);

    // Check correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(host_ref[i] - host_out[i]) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": CPU = " << host_ref[i]
                      << ", GPU = " << host_out[i] << '\n';
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✅ Output matches CPU reference for N = " << N << '\n';
    } else {
        std::cout << "❌ Output mismatch\n";
    }

    cudaFree(d_out);
    return 0;
}

