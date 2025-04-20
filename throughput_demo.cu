#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

/*

  mimic calculation
  Ex[tid] = Cax[tid]*Ex[tid] + Cbx[tid] * (Hz[tid] + Hy[tid] + Jx[tid]) * dx; 
  Ey[tid] = Cay[tid]*Ey[tid] + Cby[tid] * (Hx[tid] + Hz[tid] + Jy[tid]) * dx; 
  Ez[tid] = Caz[tid]*Ez[tid] + Cbz[tid] * (Hx[tid] + Hz[tid] + Jz[tid]) * dx; 

*/

// handle errors in CUDA call
#define CUDACHECK(call)                                                        \
{                                                                          \
   const cudaError_t error = call;                                         \
   if (error != cudaSuccess)                                               \
   {                                                                       \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
      exit(1);                                                             \
   }                                                                       \
} (void)0  // Ensures a semicolon is required after the macro call.

// number of threads in each dimension
#define NTX 32
#define NTY 4
#define NTZ 4

// length of tile in each dimension
// could be same as NTX, could be not, let's see
#define BLX 32
#define BLY 4
#define BLZ 4 

#define BLOCK_SIZE (NTX * NTY * NTZ)

__global__ void update_naive(float* Ex, float* Ey, float* Ez,
                             float* Hx, float* Hy, float* Hz,
                             float* Cax, float* Cay, float* Caz,
                             float* Cbx, float* Cby, float* Cbz,
                             float* Jx, float* Jy, float* Jz,
                             float dx,
                             size_t length
) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < length) {
    Ex[tid] = Cax[tid]*Ex[tid] + Cbx[tid] * (Hz[tid] + Hy[tid] + Jx[tid]) * dx; 
    Ey[tid] = Cay[tid]*Ey[tid] + Cby[tid] * (Hx[tid] + Hz[tid] + Jy[tid]) * dx; 
    Ez[tid] = Caz[tid]*Ey[tid] + Cbz[tid] * (Hx[tid] + Hy[tid] + Jz[tid]) * dx; 
  }

}

__global__ void update_tiling_one_to_one(float* Ex, float* Ey, float* Ez,
                                         float* Hx, float* Hy, float* Hz,
                                         float* Cax, float* Cay, float* Caz,
                                         float* Cbx, float* Cby, float* Cbz,
                                         float* Jx, float* Jy, float* Jz,
                                         float dx,
                                         size_t length) {

}

void sequential(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                std::vector<float> Hx, std::vector<float> Hy, std::vector<float> Hz,
                std::vector<float> Cax, std::vector<float> Cay, std::vector<float> Caz,
                std::vector<float> Cbx, std::vector<float> Cby, std::vector<float> Cbz,
                std::vector<float> Jx, std::vector<float> Jy, std::vector<float> Jz,
                float dx,
                int timesteps,
                int length
) {

  auto start = std::chrono::steady_clock::now();
  
  for(int t=0; t<100; t++) {
    for(int i=0; i<length; i++) {
      Ex[i] = Cax[i]*Ex[i] + Cbx[i] * (Hz[i] + Hy[i] + Jx[i]) * dx; 
      Ey[i] = Cay[i]*Ey[i] + Cby[i] * (Hx[i] + Hz[i] + Jy[i]) * dx; 
      Ez[i] = Caz[i]*Ey[i] + Cbz[i] * (Hx[i] + Hy[i] + Jz[i]) * dx; 
    }
  }

  auto end = std::chrono::steady_clock::now();
  size_t seq_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "seq runtime: " << seq_runtime << "us\n";
}

void gpu(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
         std::vector<float> Hx, std::vector<float> Hy, std::vector<float> Hz,
         std::vector<float> Cax, std::vector<float> Cay, std::vector<float> Caz,
         std::vector<float> Cbx, std::vector<float> Cby, std::vector<float> Cbz,
         std::vector<float> Jx, std::vector<float> Jy, std::vector<float> Jz,
         float dx,
         size_t length
) {

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;

  CUDACHECK(cudaMalloc(&Ex_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ey_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ez_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * length));

  cudaMemcpy(Ex_d, Ex.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Ey_d, Ey.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Ez_d, Ez.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Hx_d, Hx.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Hy_d, Hy.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Hz_d, Hz.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Jx_d, Jx.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Jy_d, Jy.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Jz_d, Jz.data(), sizeof(float) * length, cudaMemcpyHostToDevice);


  cudaMemcpy(Cax_d, Cax.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Cay_d, Cay.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Caz_d, Caz.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbx_d, Cbx.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Cby_d, Cby.data(), sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbz_d, Cbz.data(), sizeof(float) * length, cudaMemcpyHostToDevice);

  size_t grid_size = (length + BLOCK_SIZE - 1)/BLOCK_SIZE;

  auto start = std::chrono::steady_clock::now();

  for(size_t t=0; t<100; t++) {
    update_naive<<<grid_size, BLOCK_SIZE>>>(Ex_d, Ey_d, Ez_d,
                                      Hx_d, Hy_d, Hz_d,
                                      Cax_d, Cay_d, Caz_d,
                                      Cbx_d, Cby_d, Cbz_d,
                                      Jx_d, Jy_d, Jz_d,
                                      dx,
                                      length 
                                     );
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  size_t gpu_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "gpu runtime(excluding memcpy): " << gpu_runtime << "us\n";

  cudaMemcpy(Ex.data(), Ex_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey.data(), Ey_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez.data(), Ez_d, sizeof(float) * length, cudaMemcpyDeviceToHost);

  cudaFree(Ex_d);
  cudaFree(Ey_d);
  cudaFree(Ez_d);
  cudaFree(Hx_d);
  cudaFree(Hy_d);
  cudaFree(Hz_d);
  cudaFree(Cax_d);
  cudaFree(Cay_d);
  cudaFree(Caz_d);
  cudaFree(Cbx_d);
  cudaFree(Cby_d);
  cudaFree(Cbz_d);
  cudaFree(Jx_d);
  cudaFree(Jy_d);
  cudaFree(Jz_d);
}

void gpu_tiling_one_to_one(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                           std::vector<float> Hx, std::vector<float> Hy, std::vector<float> Hz,
                           std::vector<float> Cax, std::vector<float> Cay, std::vector<float> Caz,
                           std::vector<float> Cbx, std::vector<float> Cby, std::vector<float> Cbz,
                           std::vector<float> Jx, std::vector<float> Jy, std::vector<float> Jz,
                           float dx,
                           size_t length
) {

}

int main(int argc, char* argv[]) {

  if(argc != 5) {
    std::cerr << "BLX = " << BLX << ", BLY = " << BLY << ", BLZ = " << BLZ << "\n"; 
    std::cerr << "Nx = Tx * BLX\n";
    std::cerr << "usage: ./a.out Tx Ty Tz timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  float dx = 0.01;

  int Tx = std::atoi(argv[1]);
  int Ty = std::atoi(argv[2]);
  int Tz = std::atoi(argv[3]);
  int timesteps = std::atoi(argv[4]);

  int Nx = Tx * BLX;
  int Ny = Ty * BLY;
  int Nz = Tz * BLZ; 

  int array_size = Nx * Ny * Nz;
  std::vector<float> Cax(array_size, 1.0);
  std::vector<float> Cay(array_size, 1.0);
  std::vector<float> Caz(array_size, 1.0);
  std::vector<float> Cbx(array_size, 1.0);
  std::vector<float> Cby(array_size, 1.0);
  std::vector<float> Cbz(array_size, 1.0);

  std::vector<float> Jx(array_size, 1.0);
  std::vector<float> Jy(array_size, 1.0);
  std::vector<float> Jz(array_size, 1.0);
  
  std::vector<float> Hx(array_size, 1.0);
  std::vector<float> Hy(array_size, 1.0);
  std::vector<float> Hz(array_size, 1.0);

  std::vector<float> Ex_cpu_single(array_size, 1.0);
  std::vector<float> Ey_cpu_single(array_size, 1.0);
  std::vector<float> Ez_cpu_single(array_size, 1.0);
  std::vector<float> Ex_gpu_naive(array_size, 1.0);
  std::vector<float> Ey_gpu_naive(array_size, 1.0);
  std::vector<float> Ez_gpu_naive(array_size, 1.0);

  sequential(Ex_cpu_single, Ey_cpu_single, Ez_cpu_single, 
             Hx, Hy, Hz,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Jx, Jy, Jz,
             dx,
             timesteps,
             array_size);

  gpu(Ex_gpu_naive, Ey_gpu_naive, Ez_gpu_naive, 
      Hx, Hy, Hz,
      Cax, Cay, Caz,
      Cbx, Cby, Cbz,
      Jx, Jy, Jz,
      dx,
      Nx*Ny*Nz);

  bool correct = true;
  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_naive[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_naive[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_naive[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  if(!correct) {
    std::cout << "results incorrect!\n";
  }

  return 0;
}

























