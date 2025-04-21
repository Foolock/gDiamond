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

#define BLX 11 
#define MOUNTAIN BLX // number of elements (E) at mountain bottom
#define VALLEY (BLX - 2 * (BLT - 1) - 1) // number of elements (E) at valley bottom
#define LEFT_PAD BLT
#define RIGHT_PAD BLT

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

void diamond_tiling_thread_idling(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez, 
                                  std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                  const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                                  const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                                  const std::vector<float>& Dax, const std::vector<float>& Day, const std::vector<float>& Daz,
                                  const std::vector<float>& Dbx, const std::vector<float>& Dby, const std::vector<float>& Dbz,
                                  const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                  const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                  float dx,
                                  int timesteps,
                                  int Nx) {

  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;
 
  std::vector<float> Hx_pad(Nx_pad, 1);
  std::vector<float> Hy_pad(Nx_pad, 1);
  std::vector<float> Hz_pad(Nx_pad, 1);
  std::vector<float> Ex_pad(Nx_pad, 1);
  std::vector<float> Ey_pad(Nx_pad, 1);
  std::vector<float> Ez_pad(Nx_pad, 1);



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
  
  std::vector<float> Hx_cpu_single(Nx, 1);
  std::vector<float> Hy_cpu_single(Nx, 1);
  std::vector<float> Hz_cpu_single(Nx, 1);
  std::vector<float> Ex_cpu_single(Nx, 1);
  std::vector<float> Ey_cpu_single(Nx, 1);
  std::vector<float> Ez_cpu_single(Nx, 1);

  sequential(Ex_cpu_single, Ey_cpu_single, Ez_cpu_single, 
             Hx_cpu_single, Hy_cpu_single, Hz_cpu_single,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Dax, Day, Daz,
             Dbx, Dby, Dbz,
             Jx, Jy, Jz,
             Mx, My, Mz,
             dx,
             timesteps,
             Nx);

}







