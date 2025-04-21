#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

/*

  1-D stencil calculation

  E(x, t) = f(H(x, t-1), H(x-1, t))
  H(x, t) = f(E(x, t), E(x+1, t))

  Ex[tid] = Cax[tid]*Ex[tid] + Cbx[tid] * (Hz[tid] + Hy[tid - 1] + Jx[tid]) * dx; 
  Ey[tid] = Cay[tid]*Ey[tid] + Cby[tid] * (Hx[tid] + Hz[tid - 1] + Jy[tid]) * dx; 
  Ez[tid] = Caz[tid]*Ez[tid] + Cbz[tid] * (Hx[tid - 1] + Hz[tid] + Jz[tid]) * dx; 

  Hx[tid] = Dax[tid]*Hx[tid] + Dbx[tid] * (Ez[tid] + Ey[tid + 1] + Mx[tid]) * dx;
  Hy[tid] = Day[tid]*Hy[tid] + Dby[tid] * (Ex[tid] + Ez[tid + 1] + My[tid]) * dx;
  Hz[tid] = Daz[tid]*Hz[tid] + Dbz[tid] * (Ex[tid + 1] + Ey[tid] + Mz[tid]) * dx;

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

// number of time steps involved in one tile
#define BLT 4

#define BLX 512 
#define MOUNTAIN BLX // number of elements (E) at mountain bottom
#define VALLEY (BLX - 2 * (BLT - 1) - 1) // number of elements (E) at valley bottom
#define LEFT_PAD BLT
#define RIGHT_PAD BLT

#define SHX (BLX + 1)

void sequential(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                const std::vector<float>& Dax, const std::vector<float>& Day, const std::vector<float>& Daz,
                const std::vector<float>& Dbx, const std::vector<float>& Dby, const std::vector<float>& Dbz,
                const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                float dx,
                int timesteps,
                int Nx 
) {

  auto start = std::chrono::steady_clock::now();
  
  for(int t=0; t<timesteps; t++) {

    // update E
    for(int i=1; i<=Nx-2; i++) {
      Ex[i] = Cax[i]*Ex[i] + Cbx[i] * (Hz[i] + Hy[i - 1] + Jx[i]) * dx; 
      Ey[i] = Cay[i]*Ey[i] + Cby[i] * (Hx[i] + Hz[i - 1] + Jy[i]) * dx; 
      Ez[i] = Caz[i]*Ez[i] + Cbz[i] * (Hx[i - 1] + Hz[i] + Jz[i]) * dx; 
    }

    // update H
    for(int i=1; i<=Nx-2; i++) {
      Hx[i] = Dax[i]*Hx[i] + Dbx[i] * (Ez[i] + Ey[i + 1] + Mx[i]) * dx;
      Hy[i] = Day[i]*Hy[i] + Dby[i] * (Ex[i] + Ez[i + 1] + My[i]) * dx;
      Hz[i] = Daz[i]*Hz[i] + Dbz[i] * (Ex[i + 1] + Ey[i] + Mz[i]) * dx;
    }
  }

  auto end = std::chrono::steady_clock::now();
  size_t seq_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "seq runtime: " << seq_runtime << "us\n";
}

void diamond_tiling_thread_idling_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                                      std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                      const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                                      const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Day, const std::vector<float>& Daz,
                                      const std::vector<float>& Dbx, const std::vector<float>& Dby, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx,
                                      int timesteps,
                                      int Nx,
                                      int Tx) {

  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;
 
  std::vector<float> Hx_pad(Nx_pad, 1);
  std::vector<float> Hy_pad(Nx_pad, 1);
  std::vector<float> Hz_pad(Nx_pad, 1);
  std::vector<float> Ex_pad(Nx_pad, 1);
  std::vector<float> Ey_pad(Nx_pad, 1);
  std::vector<float> Ez_pad(Nx_pad, 1);

  // tiling paramemters
  int xx_num_m = Tx + 1;
  int xx_num_v = xx_num_m;
  std::vector<int> xx_heads_m(xx_num_m, 0); // head indices of mountains
  std::vector<int> xx_heads_v(xx_num_v, 0); // head indices of valleys

  for(int index=0; index<xx_num_m; index++) {
    xx_heads_m[index] = (index == 0)? 1 :
                               xx_heads_m[index-1] + (MOUNTAIN + VALLEY);
  }

  for(int index=0; index<xx_num_v; index++) {
    xx_heads_v[index] = (index == 0)? LEFT_PAD + VALLEY : 
                             xx_heads_v[index-1] + (MOUNTAIN + VALLEY);
  }

  std::cout << "xx_heads_m = ";
  for(const auto& data : xx_heads_m) {
    std::cout << data << " ";
  } 
  std::cout << "\n";

  std::cout << "xx_heads_v = ";
  for(const auto& data : xx_heads_v) {
    std::cout << data << " ";
  } 
  std::cout << "\n";

  int block_size = BLX;
  int grid_size;
  for(int tt=0; tt<timesteps/BLT; tt++) {
    // phase 1. mountain
    grid_size = xx_num_m;
    for(int xx=0; xx<grid_size; xx++) {
      
      // declare shared memory
      float Hx_pad_shmem[SHX];
      float Hy_pad_shmem[SHX];
      float Hz_pad_shmem[SHX];
      float Ex_pad_shmem[SHX];
      float Ey_pad_shmem[SHX];
      float Ez_pad_shmem[SHX];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        int shared_idx = tid + 1; // in mountain, SHX left shift one than BLX 
        int global_idx = xx_heads_m[xx] + tid;
        Hx_pad_shmem[shared_idx] = Hx_pad[global_idx];
        Hy_pad_shmem[shared_idx] = Hy_pad[global_idx];
        Hz_pad_shmem[shared_idx] = Hz_pad[global_idx];
        Ex_pad_shmem[shared_idx] = Ex_pad[global_idx];
        Ey_pad_shmem[shared_idx] = Ey_pad[global_idx];
        Ez_pad_shmem[shared_idx] = Ez_pad[global_idx];
        
        if(tid == 0) {
          Hx_pad_shmem[shared_idx - 1] = Hx_pad[global_idx - 1];
          Hy_pad_shmem[shared_idx - 1] = Hy_pad[global_idx - 1];
          Hz_pad_shmem[shared_idx - 1] = Hz_pad[global_idx - 1];
          Ex_pad_shmem[shared_idx - 1] = Ex_pad[global_idx - 1];
          Ey_pad_shmem[shared_idx - 1] = Ey_pad[global_idx - 1];
          Ez_pad_shmem[shared_idx - 1] = Ez_pad[global_idx - 1];
        }
      }

      // calculation
      for(int t=0; t<BLT; t++) {

        int calE_head = xx_heads_m[xx] + t;
        int calE_tail = xx_heads_m[xx] + BLX - 1 - t;
        int calH_head = calE_head;
        int calH_tail = calE_tail - 1;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          int global_idx = xx_heads_m[xx] + tid;
          int shared_idx = tid + 1;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
             global_idx >= calE_head && global_idx <= calE_tail) {

            Ex_pad_shmem[shared_idx] = Cax[shared_idx]*Ex_pad_shmem[shared_idx] + 
                                       Cbx[shared_idx] * (Hz_pad_shmem[shared_idx] + Hy_pad_shmem[shared_idx - 1] + Jx[shared_idx]) * dx; 
            Ey_pad_shmem[shared_idx] = Cay[shared_idx]*Ey_pad_shmem[shared_idx] + 
                                       Cby[shared_idx] * (Hx_pad_shmem[shared_idx] + Hz_pad_shmem[shared_idx - 1] + Jy[shared_idx]) * dx; 
            Ez_pad_shmem[shared_idx] = Caz[shared_idx]*Ez_pad_shmem[shared_idx] + 
                                       Cbz[shared_idx] * (Hx_pad_shmem[shared_idx - 1] + Hz_pad_shmem[shared_idx] + Jz[shared_idx]) * dx; 

          }
        }

        // update H 
        for(int tid=0; tid<block_size; tid++) {
          int global_idx = xx_heads_m[xx] + tid;
          int shared_idx = tid + 1;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
             global_idx >= calH_head && global_idx <= calH_tail) {

            Hx_pad_shmem[shared_idx] = Dax[shared_idx]*Hx_pad_shmem[shared_idx] + 
                                       Dbx[shared_idx] * (Ez_pad_shmem[shared_idx] + Ey_pad_shmem[shared_idx + 1] + Mx[shared_idx]) * dx;
            Hy_pad_shmem[shared_idx] = Day[shared_idx]*Hy_pad_shmem[shared_idx] + 
                                       Dby[shared_idx] * (Ex_pad_shmem[shared_idx] + Ez_pad_shmem[shared_idx + 1] + My[shared_idx]) * dx;
            Hz_pad_shmem[shared_idx] = Daz[shared_idx]*Hz_pad_shmem[shared_idx] + 
                                       Dbz[shared_idx] * (Ex_pad_shmem[shared_idx + 1] + Ey_pad_shmem[shared_idx] + Mz[shared_idx]) * dx;

          }
        }
      }

      // store back to global memory
      int storeE_head = xx_heads_m[xx];
      int storeE_tail = storeE_head + BLX - 1;
      int storeH_head = storeE_head;
      int storeH_tail = storeE_tail - 1;
      for(int tid=0; tid<block_size; tid++) {
        int shared_idx = tid + 1; // in mountain, SHX left shift one than BLX 
        int global_idx = xx_heads_m[xx] + tid;

        // store E
        if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
           global_idx >= storeE_head && global_idx <= storeE_tail) {

          Ex_pad[global_idx] = Ex_pad_shmem[shared_idx];
          Ey_pad[global_idx] = Ey_pad_shmem[shared_idx];
          Ez_pad[global_idx] = Ez_pad_shmem[shared_idx];

        }

        // store H
        if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
           global_idx >= storeH_head && global_idx <= storeH_tail) {

          Hx_pad[global_idx] = Hx_pad_shmem[shared_idx];
          Hy_pad[global_idx] = Hy_pad_shmem[shared_idx];
          Hz_pad[global_idx] = Hz_pad_shmem[shared_idx];

        }
      }

    }

    // phase 2. valley 
    grid_size = xx_num_v;
    for(int xx=0; xx<grid_size; xx++) {
 
      // declare shared memory
      float Hx_pad_shmem[SHX];
      float Hy_pad_shmem[SHX];
      float Hz_pad_shmem[SHX];
      float Ex_pad_shmem[SHX];
      float Ey_pad_shmem[SHX];
      float Ez_pad_shmem[SHX];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        int shared_idx = tid; // in valley, SHX starts in same line as BLX 
        int global_idx = xx_heads_v[xx] + tid;
        Hx_pad_shmem[shared_idx] = Hx_pad[global_idx];
        Hy_pad_shmem[shared_idx] = Hy_pad[global_idx];
        Hz_pad_shmem[shared_idx] = Hz_pad[global_idx];
        Ex_pad_shmem[shared_idx] = Ex_pad[global_idx];
        Ey_pad_shmem[shared_idx] = Ey_pad[global_idx];
        Ez_pad_shmem[shared_idx] = Ez_pad[global_idx];
        
        if(tid == BLX - 1) {
          Hx_pad_shmem[shared_idx + 1] = Hx_pad[global_idx + 1];
          Hy_pad_shmem[shared_idx + 1] = Hy_pad[global_idx + 1];
          Hz_pad_shmem[shared_idx + 1] = Hz_pad[global_idx + 1];
          Ex_pad_shmem[shared_idx + 1] = Ex_pad[global_idx + 1];
          Ey_pad_shmem[shared_idx + 1] = Ey_pad[global_idx + 1];
          Ez_pad_shmem[shared_idx + 1] = Ez_pad[global_idx + 1];
        }
      }

      // calculation
      for(int t=0; t<BLT; t++) {

        int calE_head = xx_heads_v[xx] + BLT - t;
        int calE_tail = xx_heads_v[xx] + BLX - 1 - (BLT - t -1);
        int calH_head = calE_head - 1;
        int calH_tail = calE_tail;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          int global_idx = xx_heads_v[xx] + tid;
          int shared_idx = tid;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
             global_idx >= calE_head && global_idx <= calE_tail) {

            Ex_pad_shmem[shared_idx] = Cax[shared_idx]*Ex_pad_shmem[shared_idx] + 
                                       Cbx[shared_idx] * (Hz_pad_shmem[shared_idx] + Hy_pad_shmem[shared_idx - 1] + Jx[shared_idx]) * dx; 
            Ey_pad_shmem[shared_idx] = Cay[shared_idx]*Ey_pad_shmem[shared_idx] + 
                                       Cby[shared_idx] * (Hx_pad_shmem[shared_idx] + Hz_pad_shmem[shared_idx - 1] + Jy[shared_idx]) * dx; 
            Ez_pad_shmem[shared_idx] = Caz[shared_idx]*Ez_pad_shmem[shared_idx] + 
                                       Cbz[shared_idx] * (Hx_pad_shmem[shared_idx - 1] + Hz_pad_shmem[shared_idx] + Jz[shared_idx]) * dx; 

          }
        }

        // update H 
        for(int tid=0; tid<block_size; tid++) {
          int global_idx = xx_heads_v[xx] + tid;
          int shared_idx = tid;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
             global_idx >= calH_head && global_idx <= calH_tail) {

            Hx_pad_shmem[shared_idx] = Dax[shared_idx]*Hx_pad_shmem[shared_idx] + 
                                       Dbx[shared_idx] * (Ez_pad_shmem[shared_idx] + Ey_pad_shmem[shared_idx + 1] + Mx[shared_idx]) * dx;
            Hy_pad_shmem[shared_idx] = Day[shared_idx]*Hy_pad_shmem[shared_idx] + 
                                       Dby[shared_idx] * (Ex_pad_shmem[shared_idx] + Ez_pad_shmem[shared_idx + 1] + My[shared_idx]) * dx;
            Hz_pad_shmem[shared_idx] = Daz[shared_idx]*Hz_pad_shmem[shared_idx] + 
                                       Dbz[shared_idx] * (Ex_pad_shmem[shared_idx + 1] + Ey_pad_shmem[shared_idx] + Mz[shared_idx]) * dx;

          }
        }
      }

      // store back to global memory
      int storeH_head = xx_heads_v[xx];
      int storeH_tail = storeH_head + BLX - 1;
      int storeE_head = storeH_head + 1;
      int storeE_tail = storeH_tail;
      for(int tid=0; tid<block_size; tid++) {
        int shared_idx = tid; // in valley, SHX starts in same line as BLX 
        int global_idx = xx_heads_v[xx] + tid;

        // store E
        if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
           global_idx >= storeE_head && global_idx <= storeE_tail) {

          Ex_pad[global_idx] = Ex_pad_shmem[shared_idx];
          Ey_pad[global_idx] = Ey_pad_shmem[shared_idx];
          Ez_pad[global_idx] = Ez_pad_shmem[shared_idx];

        }

        // store H
        if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
           global_idx >= storeH_head && global_idx <= storeH_tail) {

          Hx_pad[global_idx] = Hx_pad_shmem[shared_idx];
          Hy_pad[global_idx] = Hy_pad_shmem[shared_idx];
          Hz_pad[global_idx] = Hz_pad_shmem[shared_idx];

        }
      }
    }
  
  }

  // extract data from padded array
  for(int index=0; index<Nx; index++) {
    Ex[index] = Ex_pad[index + LEFT_PAD];
    Ey[index] = Ey_pad[index + LEFT_PAD];
    Ez[index] = Ez_pad[index + LEFT_PAD];
    Hx[index] = Hx_pad[index + LEFT_PAD];
    Hy[index] = Hy_pad[index + LEFT_PAD];
    Hz[index] = Hz_pad[index + LEFT_PAD];
  }

}

__global__ void update_mountain(float* Ex_pad, float* Ey_pad, float* Ez_pad, 
                                float* Hx_pad, float* Hy_pad, float* Hz_pad,
                                float* Cax, float* Cay, float* Caz,
                                float* Cbx, float* Cby, float* Cbz,
                                float* Dax, float* Day, float* Daz,
                                float* Dbx, float* Dby, float* Dbz,
                                float* Jx, float* Jy, float* Jz,
                                float* Mx, float* My, float* Mz,
                                float dx,
                                int* xx_heads_m, 
                                int Nx,
                                int Tx) {

  const unsigned int tid = threadIdx.x; 
  const unsigned int xx = blockIdx.x; // map a tile to a block
  const unsigned int shared_idx = tid + 1; // in mountain, SHX left shift one than BLX 
  const unsigned int global_idx = xx_heads_m[xx] + tid;

  // declare shared memory
  __shared__ float Hx_pad_shmem[SHX];
  __shared__ float Hy_pad_shmem[SHX];
  __shared__ float Hz_pad_shmem[SHX];
  __shared__ float Ex_pad_shmem[SHX];
  __shared__ float Ey_pad_shmem[SHX];
  __shared__ float Ez_pad_shmem[SHX];

  // load shared memory
  Hx_pad_shmem[shared_idx] = Hx_pad[global_idx];
  Hy_pad_shmem[shared_idx] = Hy_pad[global_idx];
  Hz_pad_shmem[shared_idx] = Hz_pad[global_idx];
  Ex_pad_shmem[shared_idx] = Ex_pad[global_idx];
  Ey_pad_shmem[shared_idx] = Ey_pad[global_idx];
  Ez_pad_shmem[shared_idx] = Ez_pad[global_idx];
  if(tid == 0) {
    Hx_pad_shmem[shared_idx - 1] = Hx_pad[global_idx - 1];
    Hy_pad_shmem[shared_idx - 1] = Hy_pad[global_idx - 1];
    Hz_pad_shmem[shared_idx - 1] = Hz_pad[global_idx - 1];
    Ex_pad_shmem[shared_idx - 1] = Ex_pad[global_idx - 1];
    Ey_pad_shmem[shared_idx - 1] = Ey_pad[global_idx - 1];
    Ez_pad_shmem[shared_idx - 1] = Ez_pad[global_idx - 1];
  }

  // calculation
  for(int t=0; t<BLT; t++) {

    const int calE_head = xx_heads_m[xx] + t;
    const int calE_tail = xx_heads_m[xx] + BLX - 1 - t;
    const int calH_head = calE_head;
    const int calH_tail = calE_tail - 1;

    // update E
    if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
       global_idx >= calE_head && global_idx <= calE_tail) {

      Ex_pad_shmem[shared_idx] = Cax[shared_idx]*Ex_pad_shmem[shared_idx] + 
                                 Cbx[shared_idx] * (Hz_pad_shmem[shared_idx] + Hy_pad_shmem[shared_idx - 1] + Jx[shared_idx]) * dx; 
      Ey_pad_shmem[shared_idx] = Cay[shared_idx]*Ey_pad_shmem[shared_idx] + 
                                 Cby[shared_idx] * (Hx_pad_shmem[shared_idx] + Hz_pad_shmem[shared_idx - 1] + Jy[shared_idx]) * dx; 
      Ez_pad_shmem[shared_idx] = Caz[shared_idx]*Ez_pad_shmem[shared_idx] + 
                                 Cbz[shared_idx] * (Hx_pad_shmem[shared_idx - 1] + Hz_pad_shmem[shared_idx] + Jz[shared_idx]) * dx; 

    }

    __syncthreads();

    // update H
    if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
       global_idx >= calH_head && global_idx <= calH_tail) {

      Hx_pad_shmem[shared_idx] = Dax[shared_idx]*Hx_pad_shmem[shared_idx] +
                                 Dbx[shared_idx] * (Ez_pad_shmem[shared_idx] + Ey_pad_shmem[shared_idx + 1] + Mx[shared_idx]) * dx;
      Hy_pad_shmem[shared_idx] = Day[shared_idx]*Hy_pad_shmem[shared_idx] +
                                 Dby[shared_idx] * (Ex_pad_shmem[shared_idx] + Ez_pad_shmem[shared_idx + 1] + My[shared_idx]) * dx;
      Hz_pad_shmem[shared_idx] = Daz[shared_idx]*Hz_pad_shmem[shared_idx] +
                                 Dbz[shared_idx] * (Ex_pad_shmem[shared_idx + 1] + Ey_pad_shmem[shared_idx] + Mz[shared_idx]) * dx;

    }

    __syncthreads();
  }

  // store back to global memory
  const int storeE_head = xx_heads_m[xx];
  const int storeE_tail = storeE_head + BLX - 1;
  const int storeH_head = storeE_head;
  const int storeH_tail = storeE_tail - 1;

  // store E
  if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
     global_idx >= storeE_head && global_idx <= storeE_tail) {

    Ex_pad[global_idx] = Ex_pad_shmem[shared_idx];
    Ey_pad[global_idx] = Ey_pad_shmem[shared_idx];
    Ez_pad[global_idx] = Ez_pad_shmem[shared_idx];

  }

  // store H
  if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
     global_idx >= storeH_head && global_idx <= storeH_tail) {

    Hx_pad[global_idx] = Hx_pad_shmem[shared_idx];
    Hy_pad[global_idx] = Hy_pad_shmem[shared_idx];
    Hz_pad[global_idx] = Hz_pad_shmem[shared_idx];

  }
}

__global__ void update_valley(float* Ex_pad, float* Ey_pad, float* Ez_pad, 
                              float* Hx_pad, float* Hy_pad, float* Hz_pad,
                              float* Cax, float* Cay, float* Caz,
                              float* Cbx, float* Cby, float* Cbz,
                              float* Dax, float* Day, float* Daz,
                              float* Dbx, float* Dby, float* Dbz,
                              float* Jx, float* Jy, float* Jz,
                              float* Mx, float* My, float* Mz,
                              float dx,
                              int *xx_heads_v,
                              int Nx,
                              int Tx) {

  const unsigned int tid = threadIdx.x; 
  const unsigned int xx = blockIdx.x; // map a tile to a block
  const unsigned int shared_idx = tid; // in valley, SHX starts in same line as BLX
  const unsigned int global_idx = xx_heads_v[xx] + tid;

  // declare shared memory
  __shared__ float Hx_pad_shmem[SHX];
  __shared__ float Hy_pad_shmem[SHX];
  __shared__ float Hz_pad_shmem[SHX];
  __shared__ float Ex_pad_shmem[SHX];
  __shared__ float Ey_pad_shmem[SHX];
  __shared__ float Ez_pad_shmem[SHX];

  // load shared memory
  Hx_pad_shmem[shared_idx] = Hx_pad[global_idx];
  Hy_pad_shmem[shared_idx] = Hy_pad[global_idx];
  Hz_pad_shmem[shared_idx] = Hz_pad[global_idx];
  Ex_pad_shmem[shared_idx] = Ex_pad[global_idx];
  Ey_pad_shmem[shared_idx] = Ey_pad[global_idx];
  Ez_pad_shmem[shared_idx] = Ez_pad[global_idx];
  if(tid == BLX - 1) {
    Hx_pad_shmem[shared_idx + 1] = Hx_pad[global_idx + 1];
    Hy_pad_shmem[shared_idx + 1] = Hy_pad[global_idx + 1];
    Hz_pad_shmem[shared_idx + 1] = Hz_pad[global_idx + 1];
    Ex_pad_shmem[shared_idx + 1] = Ex_pad[global_idx + 1];
    Ey_pad_shmem[shared_idx + 1] = Ey_pad[global_idx + 1];
    Ez_pad_shmem[shared_idx + 1] = Ez_pad[global_idx + 1];
  }

  // calculation
  for(int t=0; t<BLT; t++) {

    const int calE_head = xx_heads_v[xx] + BLT - t;
    const int calE_tail = xx_heads_v[xx] + BLX - 1 - (BLT - t -1);
    const int calH_head = calE_head - 1;
    const int calH_tail = calE_tail;

    // update E
    if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
       global_idx >= calE_head && global_idx <= calE_tail) {

      Ex_pad_shmem[shared_idx] = Cax[shared_idx]*Ex_pad_shmem[shared_idx] + 
                                 Cbx[shared_idx] * (Hz_pad_shmem[shared_idx] + Hy_pad_shmem[shared_idx - 1] + Jx[shared_idx]) * dx; 
      Ey_pad_shmem[shared_idx] = Cay[shared_idx]*Ey_pad_shmem[shared_idx] + 
                                 Cby[shared_idx] * (Hx_pad_shmem[shared_idx] + Hz_pad_shmem[shared_idx - 1] + Jy[shared_idx]) * dx; 
      Ez_pad_shmem[shared_idx] = Caz[shared_idx]*Ez_pad_shmem[shared_idx] + 
                                 Cbz[shared_idx] * (Hx_pad_shmem[shared_idx - 1] + Hz_pad_shmem[shared_idx] + Jz[shared_idx]) * dx; 

    }

    __syncthreads();

    // update H
    if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
       global_idx >= calH_head && global_idx <= calH_tail) {

      Hx_pad_shmem[shared_idx] = Dax[shared_idx]*Hx_pad_shmem[shared_idx] + 
                                 Dbx[shared_idx] * (Ez_pad_shmem[shared_idx] + Ey_pad_shmem[shared_idx + 1] + Mx[shared_idx]) * dx;
      Hy_pad_shmem[shared_idx] = Day[shared_idx]*Hy_pad_shmem[shared_idx] + 
                                 Dby[shared_idx] * (Ex_pad_shmem[shared_idx] + Ez_pad_shmem[shared_idx + 1] + My[shared_idx]) * dx;
      Hz_pad_shmem[shared_idx] = Daz[shared_idx]*Hz_pad_shmem[shared_idx] + 
                                 Dbz[shared_idx] * (Ex_pad_shmem[shared_idx + 1] + Ey_pad_shmem[shared_idx] + Mz[shared_idx]) * dx;

    }

    __syncthreads();
  }

  // store back to global memory
  const int storeH_head = xx_heads_v[xx];
  const int storeH_tail = storeH_head + BLX - 1;
  const int storeE_head = storeH_head + 1;
  const int storeE_tail = storeH_tail;

  // store E
  if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
     global_idx >= storeE_head && global_idx <= storeE_tail) {

    Ex_pad[global_idx] = Ex_pad_shmem[shared_idx];
    Ey_pad[global_idx] = Ey_pad_shmem[shared_idx];
    Ez_pad[global_idx] = Ez_pad_shmem[shared_idx];

  }

  // store H
  if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD &&
     global_idx >= storeH_head && global_idx <= storeH_tail) {

    Hx_pad[global_idx] = Hx_pad_shmem[shared_idx];
    Hy_pad[global_idx] = Hy_pad_shmem[shared_idx];
    Hz_pad[global_idx] = Hz_pad_shmem[shared_idx];

  }
}

void diamond_tiling_thread_idling_gpu(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                                      std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                      const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                                      const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Day, const std::vector<float>& Daz,
                                      const std::vector<float>& Dbx, const std::vector<float>& Dby, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx,
                                      int timesteps,
                                      int Nx,
                                      int Tx) {

  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;
 
  std::vector<float> Hx_pad(Nx_pad, 1);
  std::vector<float> Hy_pad(Nx_pad, 1);
  std::vector<float> Hz_pad(Nx_pad, 1);
  std::vector<float> Ex_pad(Nx_pad, 1);
  std::vector<float> Ey_pad(Nx_pad, 1);
  std::vector<float> Ez_pad(Nx_pad, 1);

  // tiling paramemters
  int xx_num_m = Tx + 1;
  int xx_num_v = xx_num_m;
  std::vector<int> xx_heads_m(xx_num_m, 0); // head indices of mountains
  std::vector<int> xx_heads_v(xx_num_v, 0); // head indices of valleys

  for(int index=0; index<xx_num_m; index++) {
    xx_heads_m[index] = (index == 0)? 1 :
                               xx_heads_m[index-1] + (MOUNTAIN + VALLEY);
  }

  for(int index=0; index<xx_num_v; index++) {
    xx_heads_v[index] = (index == 0)? LEFT_PAD + VALLEY : 
                             xx_heads_v[index-1] + (MOUNTAIN + VALLEY);
  }

  float *Ex_pad_d, *Ey_pad_d, *Ez_pad_d;
  float *Hx_pad_d, *Hy_pad_d, *Hz_pad_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Dax_d, *Day_d, *Daz_d;
  float *Dbx_d, *Dby_d, *Dbz_d;
  float *Jx_d, *Jy_d, *Jz_d;
  float *Mx_d, *My_d, *Mz_d;
  int *xx_heads_m_d, *xx_heads_v_d;

  CUDACHECK(cudaMalloc(&Ex_pad_d, sizeof(float) * Nx_pad));
  CUDACHECK(cudaMalloc(&Ey_pad_d, sizeof(float) * Nx_pad));
  CUDACHECK(cudaMalloc(&Ez_pad_d, sizeof(float) * Nx_pad));
  CUDACHECK(cudaMalloc(&Hx_pad_d, sizeof(float) * Nx_pad));
  CUDACHECK(cudaMalloc(&Hy_pad_d, sizeof(float) * Nx_pad));
  CUDACHECK(cudaMalloc(&Hz_pad_d, sizeof(float) * Nx_pad));

  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * Nx));

  CUDACHECK(cudaMalloc(&Dax_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Day_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Daz_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Dbx_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Dby_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Dbz_d, sizeof(float) * Nx));

  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Mx_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&My_d, sizeof(float) * Nx));
  CUDACHECK(cudaMalloc(&Mz_d, sizeof(float) * Nx));

  CUDACHECK(cudaMalloc(&xx_heads_m_d, sizeof(int) * xx_num_m));
  CUDACHECK(cudaMalloc(&xx_heads_v_d, sizeof(int) * xx_num_v));

  cudaMemcpy(Ex_pad_d, Ex_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);
  cudaMemcpy(Ey_pad_d, Ey_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);
  cudaMemcpy(Ez_pad_d, Ez_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);
  cudaMemcpy(Hx_pad_d, Hx_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);
  cudaMemcpy(Hy_pad_d, Hy_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);
  cudaMemcpy(Hz_pad_d, Hz_pad.data(), sizeof(float) * Nx_pad, cudaMemcpyHostToDevice);

  cudaMemcpy(Jx_d, Jx.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Jy_d, Jy.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Jz_d, Jz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Mx_d, Mx.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(My_d, My.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Mz_d, Mz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);

  cudaMemcpy(Cax_d, Cax.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Cay_d, Cay.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Caz_d, Caz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbx_d, Cbx.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Cby_d, Cby.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Cbz_d, Cbz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Dax_d, Dax.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Day_d, Day.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Daz_d, Daz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Dbx_d, Dbx.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Dby_d, Dby.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);
  cudaMemcpy(Dbz_d, Dbz.data(), sizeof(float) * Nx, cudaMemcpyHostToDevice);

  cudaMemcpy(xx_heads_m_d, xx_heads_m.data(), sizeof(int) * xx_num_m, cudaMemcpyHostToDevice);
  cudaMemcpy(xx_heads_v_d, xx_heads_v.data(), sizeof(int) * xx_num_v, cudaMemcpyHostToDevice);

  size_t block_size = BLX;
  size_t grid_size;

  auto start = std::chrono::steady_clock::now();
  
  for(int tt=0; tt<timesteps/BLT; tt++) {
    // phase 1. mountain
    grid_size = xx_num_m;
    update_mountain<<<grid_size, block_size>>>(Ex_pad_d, Ey_pad_d, Ez_pad_d, 
                                               Hx_pad_d, Hy_pad_d, Hz_pad_d,
                                               Cax_d, Cay_d, Caz_d,
                                               Cbx_d, Cby_d, Cbz_d,
                                               Dax_d, Day_d, Daz_d,
                                               Dbx_d, Dby_d, Dbz_d,
                                               Jx_d, Jy_d, Jz_d,
                                               Mx_d, My_d, Mz_d,
                                               dx,
                                               xx_heads_m_d, 
                                               Nx,
                                               Tx);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "update_mountain launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // phase 2. valley
    grid_size = xx_num_v;
    update_valley<<<grid_size, block_size>>>(Ex_pad_d, Ey_pad_d, Ez_pad_d, 
                                             Hx_pad_d, Hy_pad_d, Hz_pad_d,
                                             Cax_d, Cay_d, Caz_d,
                                             Cbx_d, Cby_d, Cbz_d,
                                             Dax_d, Day_d, Daz_d,
                                             Dbx_d, Dby_d, Dbz_d,
                                             Jx_d, Jy_d, Jz_d,
                                             Mx_d, My_d, Mz_d,
                                             dx,
                                             xx_heads_v_d, 
                                             Nx,
                                             Tx);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "update_valley launch failed: " << cudaGetErrorString(err) << std::endl;
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  size_t gpu_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "gpu diamond tiling runtime(excluding memcpy): " << gpu_runtime << "us\n";

  cudaMemcpy(Ex_pad.data(), Ex_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey_pad.data(), Ey_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez_pad.data(), Ez_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);
  cudaMemcpy(Hx_pad.data(), Hx_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);
  cudaMemcpy(Hy_pad.data(), Hy_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);
  cudaMemcpy(Hz_pad.data(), Hz_pad_d, sizeof(float) * Nx_pad, cudaMemcpyDeviceToHost);

  CUDACHECK(cudaFree(Ex_pad_d));
  CUDACHECK(cudaFree(Ey_pad_d));
  CUDACHECK(cudaFree(Ez_pad_d));
  CUDACHECK(cudaFree(Hx_pad_d));
  CUDACHECK(cudaFree(Hy_pad_d));
  CUDACHECK(cudaFree(Hz_pad_d));

  CUDACHECK(cudaFree(Cax_d));
  CUDACHECK(cudaFree(Cay_d));
  CUDACHECK(cudaFree(Caz_d));
  CUDACHECK(cudaFree(Cbx_d));
  CUDACHECK(cudaFree(Cby_d));
  CUDACHECK(cudaFree(Cbz_d));

  CUDACHECK(cudaFree(Dax_d));
  CUDACHECK(cudaFree(Day_d));
  CUDACHECK(cudaFree(Daz_d));
  CUDACHECK(cudaFree(Dbx_d));
  CUDACHECK(cudaFree(Dby_d));
  CUDACHECK(cudaFree(Dbz_d));

  CUDACHECK(cudaFree(Jx_d));
  CUDACHECK(cudaFree(Jy_d));
  CUDACHECK(cudaFree(Jz_d));
  CUDACHECK(cudaFree(Mx_d));
  CUDACHECK(cudaFree(My_d));
  CUDACHECK(cudaFree(Mz_d));

  CUDACHECK(cudaFree(xx_heads_m_d));
  CUDACHECK(cudaFree(xx_heads_v_d));

  // extract data from padded array
  for(int index=0; index<Nx; index++) {
    Ex[index] = Ex_pad[index + LEFT_PAD];
    Ey[index] = Ey_pad[index + LEFT_PAD];
    Ez[index] = Ez_pad[index + LEFT_PAD];
    Hx[index] = Hx_pad[index + LEFT_PAD];
    Hy[index] = Hy_pad[index + LEFT_PAD];
    Hz[index] = Hz_pad[index + LEFT_PAD];
  }

}

int main(int argc, char* argv[]) {

  if(argc != 3) {
    std::cerr << "MOUNTAIN = " << MOUNTAIN << ", VALLEY = " << VALLEY << "\n";
    std::cerr << "usage: ./a.out Tx timesteps\n";
    std::cerr << "Nx = Tx * (MOUNTAIN + VALLEY) + MOUNTAIN - (BLT - 1) + VALLEY\n";
    std::exit(EXIT_FAILURE);
  }

  float dx = 1;

  int Tx = std::atoi(argv[1]);
  int timesteps = std::atoi(argv[2]);
  int Nx = Tx * (MOUNTAIN + VALLEY) + MOUNTAIN - (BLT - 1) + VALLEY; 

  std::cout << "Nx = " << Nx << "\n";

  std::vector<float> Cax(Nx, 0.1);
  std::vector<float> Cay(Nx, 0.2);
  std::vector<float> Caz(Nx, 0.3);
  std::vector<float> Cbx(Nx, 0.1);
  std::vector<float> Cby(Nx, 0.1);
  std::vector<float> Cbz(Nx, 0.1);
  std::vector<float> Dax(Nx, 0.1);
  std::vector<float> Day(Nx, 0.1);
  std::vector<float> Daz(Nx, 0.1);
  std::vector<float> Dbx(Nx, 0.1);
  std::vector<float> Dby(Nx, 0.1);
  std::vector<float> Dbz(Nx, 0.1);

  std::vector<float> Jx(Nx, 0.1);
  std::vector<float> Jy(Nx, 0.1);
  std::vector<float> Jz(Nx, 0.1);
  std::vector<float> Mx(Nx, 0.1);
  std::vector<float> My(Nx, 0.1);
  std::vector<float> Mz(Nx, 0.1);
  
  std::vector<float> Hx_seq(Nx, 1);
  std::vector<float> Hy_seq(Nx, 1);
  std::vector<float> Hz_seq(Nx, 1);
  std::vector<float> Ex_seq(Nx, 1);
  std::vector<float> Ey_seq(Nx, 1);
  std::vector<float> Ez_seq(Nx, 1);

  std::vector<float> Hx_dt_seq(Nx, 1);
  std::vector<float> Hy_dt_seq(Nx, 1);
  std::vector<float> Hz_dt_seq(Nx, 1);
  std::vector<float> Ex_dt_seq(Nx, 1);
  std::vector<float> Ey_dt_seq(Nx, 1);
  std::vector<float> Ez_dt_seq(Nx, 1);

  std::vector<float> Hx_dt_gpu(Nx, 1);
  std::vector<float> Hy_dt_gpu(Nx, 1);
  std::vector<float> Hz_dt_gpu(Nx, 1);
  std::vector<float> Ex_dt_gpu(Nx, 1);
  std::vector<float> Ey_dt_gpu(Nx, 1);
  std::vector<float> Ez_dt_gpu(Nx, 1);

  sequential(Ex_seq, Ey_seq, Ez_seq, 
             Hx_seq, Hy_seq, Hz_seq,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Dax, Day, Daz,
             Dbx, Dby, Dbz,
             Jx, Jy, Jz,
             Mx, My, Mz,
             dx,
             timesteps,
             Nx);

  diamond_tiling_thread_idling_seq(Ex_dt_seq, Ey_dt_seq, Ez_dt_seq, 
                                   Hx_dt_seq, Hy_dt_seq, Hz_dt_seq,
                                   Cax, Cay, Caz,
                                   Cbx, Cby, Cbz,
                                   Dax, Day, Daz,
                                   Dbx, Dby, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   dx,
                                   timesteps,
                                   Nx,
                                   Tx);

  diamond_tiling_thread_idling_gpu(Ex_dt_gpu, Ey_dt_gpu, Ez_dt_gpu, 
                                   Hx_dt_gpu, Hy_dt_gpu, Hz_dt_gpu,
                                   Cax, Cay, Caz,
                                   Cbx, Cby, Cbz,
                                   Dax, Day, Daz,
                                   Dbx, Dby, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   dx,
                                   timesteps,
                                   Nx,
                                   Tx);

  bool correct = true;
  for(size_t i=0; i<Nx; i++) {
    if(fabs(Ex_seq[i] - Ex_dt_seq[i]) > 1e-5 ||
       fabs(Ey_seq[i] - Ey_dt_seq[i]) > 1e-5 ||
       fabs(Ez_seq[i] - Ez_dt_seq[i]) > 1e-5 ||
       fabs(Hx_seq[i] - Hx_dt_seq[i]) > 1e-5 ||
       fabs(Hy_seq[i] - Hy_dt_seq[i]) > 1e-5 ||
       fabs(Hz_seq[i] - Hz_dt_seq[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  for(size_t i=0; i<Nx; i++) {
    if(fabs(Ex_seq[i] - Ex_dt_gpu[i]) > 1e-5 ||
       fabs(Ey_seq[i] - Ey_dt_gpu[i]) > 1e-5 ||
       fabs(Ez_seq[i] - Ez_dt_gpu[i]) > 1e-5 ||
       fabs(Hx_seq[i] - Hx_dt_gpu[i]) > 1e-5 ||
       fabs(Hy_seq[i] - Hy_dt_gpu[i]) > 1e-5 ||
       fabs(Hz_seq[i] - Hz_dt_gpu[i]) > 1e-5) {
      correct = false;
      break;
    }
  }

  if(!correct) {
    std::cout << "results incorrect!\n";
  }



}







