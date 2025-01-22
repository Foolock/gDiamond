#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

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

#define BLOCK_SIZE 1024

__global__ void update(float* Ex, float* Ey, float* Ez,
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
    Ez[tid] = Caz[tid]*Ey[tid] + Cby[tid] * (Hx[tid] + Hz[tid] + Jy[tid]) * dx; 
  }

}

__global__ void check(float *Ex, 
                 size_t length
) {

  for(size_t i=0; i<length; i++) {
    printf("%f ", Ex[i]); 
  }
  printf("\n");

}

void sequential(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                std::vector<float> Hx, std::vector<float> Hy, std::vector<float> Hz,
                std::vector<float> Cax, std::vector<float> Cay, std::vector<float> Caz,
                std::vector<float> Cbx, std::vector<float> Cby, std::vector<float> Cbz,
                std::vector<float> Jx, std::vector<float> Jy, std::vector<float> Jz,
                float dx,
                size_t length
) {

  auto start = std::chrono::steady_clock::now();
  
  for(size_t t=0; t<100; t++) {
    for(size_t i=0; i<length; i++) {
      Ex[i] = Cax[i]*Ex[i] + Cbx[i] * (Hz[i] + Hy[i] + Jx[i]) * dx; 
      Ey[i] = Cay[i]*Ey[i] + Cby[i] * (Hx[i] + Hz[i] + Jy[i]) * dx; 
      Ez[i] = Caz[i]*Ey[i] + Cby[i] * (Hx[i] + Hz[i] + Jy[i]) * dx; 
    }
  }

  auto end = std::chrono::steady_clock::now();
  size_t seq_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "seq runtime: " << seq_runtime << "us\n";
}

int main() {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;

  float dx = 0.01;

  std::vector<float> Ex(Nx * Ny * Nz, 1); 
  std::vector<float> Ey(Nx * Ny * Nz, 1); 
  std::vector<float> Ez(Nx * Ny * Nz, 1); 

  std::vector<float> Ex_c(Nx * Ny * Nz, 1); 
  std::vector<float> Ey_c(Nx * Ny * Nz, 1); 
  std::vector<float> Ez_c(Nx * Ny * Nz, 1); 

  std::vector<float> Hx(Nx * Ny * Nz, 1); 
  std::vector<float> Hy(Nx * Ny * Nz, 1); 
  std::vector<float> Hz(Nx * Ny * Nz, 1); 

  std::vector<float> Cax(Nx * Ny * Nz, 1); 
  std::vector<float> Cay(Nx * Ny * Nz, 1); 
  std::vector<float> Caz(Nx * Ny * Nz, 1); 

  std::vector<float> Cbx(Nx * Ny * Nz, 1); 
  std::vector<float> Cby(Nx * Ny * Nz, 1); 
  std::vector<float> Cbz(Nx * Ny * Nz, 1); 

  std::vector<float> Jx(Nx * Ny * Nz, 1); 
  std::vector<float> Jy(Nx * Ny * Nz, 1); 
  std::vector<float> Jz(Nx * Ny * Nz, 1); 

  sequential(Ex_c, Ey_c, Ez_c, 
             Hx, Hy, Hz,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Jx, Jy, Jz,
             dx,
             Nx*Ny*Nz);

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;

  CUDACHECK(cudaMalloc(&Ex_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Ey_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Ez_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Hx_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Hy_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Hz_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * Nx * Ny * Nz));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * Nx * Ny * Nz));

  cudaMemcpy(Ex_d, Ex.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Ey_d, Ey.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Ez_d, Ez.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Hx_d, Hx.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Hy_d, Hy.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Hz_d, Hz.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Jx_d, Jx.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Jy_d, Jy.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Jz_d, Jz.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  cudaMemcpy(Cax_d, Cax.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Cay_d, Cay.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Caz_d, Caz.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbx_d, Cbx.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Cby_d, Cby.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbz_d, Cbz.data(), sizeof(float) * Nx * Ny * Nz, cudaMemcpyHostToDevice);

  size_t grid_size = (Nx*Ny*Nz + BLOCK_SIZE - 1)/BLOCK_SIZE;

  for(size_t t=0; t<100; t++) {
    update<<<grid_size, BLOCK_SIZE>>>(Ex_d, Ey_d, Ez_d,
                                      Hx_d, Hy_d, Hz_d,
                                      Cax_d, Cay_d, Caz_d,
                                      Cbx_d, Cby_d, Cbz_d,
                                      Jx_d, Jy_d, Jz_d,
                                      dx,
                                      Nx*Ny*Nz
                                     );

  }

  cudaMemcpy(Ex.data(), Ex_d, sizeof(float) * Nx * Ny * Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey.data(), Ey_d, sizeof(float) * Nx * Ny * Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez.data(), Ez_d, sizeof(float) * Nx * Ny * Nz, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  size_t gpu_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "gpu runtime: " << gpu_runtime << "us\n";

  bool correct = true;
  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_c[i] - Ex[i]) > 1e-5 ||
       fabs(Ey_c[i] - Ey[i]) > 1e-5 ||
       fabs(Ez_c[i] - Ez[i]) > 1e-5) {
      correct = false;
      break;
    }
  }
  if(!correct) {
    std::cout << "results incorrect!\n";
  }

  // std::cout << "Ex_c = \n";
  // for(size_t i=0; i<Nx*Ny*Nz; i++) {
  //   std::cout << Ex_c[i] << " ";
  // }
  // std::cout << "\n";

  // std::cout << "Ex = \n";
  // for(size_t i=0; i<Nx*Ny*Nz; i++) {
  //   std::cout << Ex[i] << " ";
  // }
  // std::cout << "\n";

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

  return 0;
}

























