#include <iostream>
#include <cuda_runtime.h>

__global__ void shared_memory_test_kernel(int *out, int shared_mem_size) {
    extern __shared__ int shared_mem[]; // Shared memory allocation

    int tid = threadIdx.x;
    if (tid < shared_mem_size / sizeof(int)) {
        shared_mem[tid] = tid;  // Fill shared memory
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < shared_mem_size / sizeof(int); i++) {
            sum += shared_mem[i];  // Simple computation to avoid optimization
        }
        out[blockIdx.x] = sum;
    }
}

void checkCudaErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int grid_size = 5000;    // Change this as needed
    int block_size = 512;  // Change this as needed
    int shared_mem_size = 48438; // Adjust this value to test limits

    int *d_out, *h_out;
    h_out = (int*)malloc(grid_size * sizeof(int));

    checkCudaErrors(cudaMalloc((void**)&d_out, grid_size * sizeof(int)), "Allocating device output memory");

    shared_memory_test_kernel<<<grid_size, block_size, shared_mem_size>>>(d_out, shared_mem_size);
    checkCudaErrors(cudaGetLastError(), "Kernel launch failed");
    checkCudaErrors(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaErrors(cudaMemcpy(h_out, d_out, grid_size * sizeof(int), cudaMemcpyDeviceToHost), "Copying output to host");

    std::cout << "Kernel execution completed. First output: " << h_out[0] << std::endl;

    free(h_out);
    cudaFree(d_out);

    return 0;
}

