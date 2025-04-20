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
                                         int* xx_heads, 
                                         int* yy_heads,
                                         int* zz_heads,
                                         int xx_num,
                                         int yy_num,
                                         int zz_num,
                                         int Nx, int Ny, int Nz) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  // Compute the tile (xx, yy, zz) that this block is responsible for
  int xx = block_id % xx_num;
  int yy = (block_id / xx_num) % yy_num;
  int zz = block_id / (xx_num * yy_num);

  // Compute local thread coordinates within the tile
  int local_x = thread_id % BLX;
  int local_y = (thread_id / BLX) % BLY;
  int local_z = thread_id / (BLX * BLY);

  // Compute global indices based on tile offsets
  int global_x = xx_heads[xx] + local_x;
  int global_y = yy_heads[yy] + local_y;
  int global_z = zz_heads[zz] + local_z;

  int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; 

  Ex[global_idx] = Cax[global_idx]*Ex[global_idx] + Cbx[global_idx] * (Hz[global_idx] + Hy[global_idx] + Jx[global_idx]) * dx; 
  Ey[global_idx] = Cay[global_idx]*Ey[global_idx] + Cby[global_idx] * (Hx[global_idx] + Hz[global_idx] + Jy[global_idx]) * dx; 
  Ez[global_idx] = Caz[global_idx]*Ey[global_idx] + Cbz[global_idx] * (Hx[global_idx] + Hy[global_idx] + Jz[global_idx]) * dx; 

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
         int timesteps,
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

  for(int t=0; t<timesteps; t++) {
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
  std::cout << "gpu naive runtime(excluding memcpy): " << gpu_runtime << "us\n";

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
                           int xx_num, 
                           int yy_num,
                           int zz_num,
                           int Nx, int Ny, int Nz,
                           int timesteps,
                           size_t length
) {

  // tiling parameters
  std::vector<int> xx_heads(xx_num, 0);
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);
  for(int i=0; i<xx_num; i++) {
    xx_heads[i] = i * BLX;
  }
  for(int i=0; i<yy_num; i++) {
    yy_heads[i] = i * BLY;
  }
  for(int i=0; i<zz_num; i++) {
    zz_heads[i] = i * BLZ;
  }

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;
  int *xx_heads_d;
  int *yy_heads_d;
  int *zz_heads_d;

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
  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * zz_num));

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

  cudaMemcpy(xx_heads_d, xx_heads.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice);
  cudaMemcpy(yy_heads_d, yy_heads.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice);
  cudaMemcpy(zz_heads_d, zz_heads.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice);

  size_t grid_size = xx_num * yy_num * zz_num;
  auto start = std::chrono::steady_clock::now();
  for(int t=0; t<timesteps; t++) {
    update_tiling_one_to_one<<<grid_size, BLOCK_SIZE>>>(Ex_d, Ey_d, Ez_d,
                                                        Hx_d, Hy_d, Hz_d,
                                                        Cax_d, Cay_d, Caz_d,
                                                        Cbx_d, Cby_d, Cbz_d,
                                                        Jx_d, Jy_d, Jz_d,
                                                        dx,
                                                        xx_heads_d, 
                                                        yy_heads_d,
                                                        zz_heads_d,
                                                        xx_num,
                                                        yy_num,
                                                        zz_num,
                                                        Nx, Ny, Nz);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  size_t gpu_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "gpu tiling one_to_one runtime(excluding memcpy): " << gpu_runtime << "us\n";

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
  cudaFree(xx_heads_d);
  cudaFree(yy_heads_d);
  cudaFree(zz_heads_d);

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
  std::vector<float> Ex_gpu_tiling_one_to_one(array_size, 1.0);
  std::vector<float> Ey_gpu_tiling_one_to_one(array_size, 1.0);
  std::vector<float> Ez_gpu_tiling_one_to_one(array_size, 1.0);

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
      timesteps,
      Nx*Ny*Nz);

  gpu_tiling_one_to_one(Ex_gpu_tiling_one_to_one, Ey_gpu_tiling_one_to_one, Ez_gpu_tiling_one_to_one, 
                        Hx, Hy, Hz,
                        Cax, Cay, Caz,
                        Cbx, Cby, Cbz,
                        Jx, Jy, Jz,
                        dx,
                        Tx, 
                        Ty,
                        Tz,
                        Nx, Ny, Nz,
                        timesteps,
                        array_size); 

  bool correct = true;
  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_naive[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_naive[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_naive[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_tiling_one_to_one[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_tiling_one_to_one[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_tiling_one_to_one[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  if(!correct) {
    std::cout << "results incorrect!\n";
  }

  return 0;
}

























