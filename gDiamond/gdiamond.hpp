#ifndef GDIAMOND_HPP 
#define GDIAMOND_HPP 

#include <iostream>
#include <random> 
#include <chrono>

namespace gdiamond {

class gDiamond {

  public:
    gDiamond(size_t Nx, size_t Ny, size_t Nz): _Nx(Nx), _Ny(Ny), _Nz(Nz),
                                               _Ex(Nx * Ny * Nz), _Ey(Nx * Ny * Nz), _Ez(Nx * Ny * Nz),
                                               _Hx(Nx * Ny * Nz), _Hy(Nx * Ny * Nz), _Hz(Nx * Ny * Nz),
                                               _Ex_seq(Nx * Ny * Nz), _Ey_seq(Nx * Ny * Nz), _Ez_seq(Nx * Ny * Nz),
                                               _Hx_seq(Nx * Ny * Nz), _Hy_seq(Nx * Ny * Nz), _Hz_seq(Nx * Ny * Nz),
                                               _Ex_gpu(Nx * Ny * Nz), _Ey_gpu(Nx * Ny * Nz), _Ez_gpu(Nx * Ny * Nz),
                                               _Hx_gpu(Nx * Ny * Nz), _Hy_gpu(Nx * Ny * Nz), _Hz_gpu(Nx * Ny * Nz),
                                               _Ex_omp(Nx * Ny * Nz), _Ey_omp(Nx * Ny * Nz), _Ez_omp(Nx * Ny * Nz),
                                               _Hx_omp(Nx * Ny * Nz), _Hy_omp(Nx * Ny * Nz), _Hz_omp(Nx * Ny * Nz),
                                               _Jx(Nx * Ny * Nz, 1), _Jy(Nx * Ny * Nz, 1), _Jz(Nx * Ny * Nz, 1),
                                               _Mx(Nx * Ny * Nz, 1), _My(Nx * Ny * Nz, 1), _Mz(Nx * Ny * Nz, 1),
                                               _Cax(Nx * Ny * Nz, 1), _Cay(Nx * Ny * Nz, 1), _Caz(Nx * Ny * Nz, 1),
                                               _Cbx(Nx * Ny * Nz, 1), _Cby(Nx * Ny * Nz, 1), _Cbz(Nx * Ny * Nz, 1),
                                               _Dax(Nx * Ny * Nz, 1), _Day(Nx * Ny * Nz, 1), _Daz(Nx * Ny * Nz, 1),
                                               _Dbx(Nx * Ny * Nz, 1), _Dby(Nx * Ny * Nz, 1), _Dbz(Nx * Ny * Nz, 1)
    {

      // fix random seed so we generate same numbers every time
      const int fixedSeed = 42;
      std::mt19937 gen(fixedSeed); // Standard mersenne_twister_engine
      std::uniform_real_distribution<> dis(0.0, 1.0); // Range [0, 1]

      // randomly fill up E and H 
      for(size_t i=0; i<Nx*Ny*Nz; i++) {
        _Ex[i] = dis(gen);
        _Ey[i] = -_Ex[i];
        _Ez[i] = _Ex[i];
        _Hx[i] = -_Ex[i] + 1;
        _Hy[i] = _Hx[i];
        _Hz[i] = -_Hx[i];
      }

      std::cout << "randomly initialize E and H, initialize J, M, Ca, Cb, Da, Db all as 1\n";
    }

    // run FDTD in cpu single thread
    void update_FDTD_seq(size_t num_timesteps);

    // run FDTD in openmp
    void update_FDTD_omp(size_t num_timesteps);

    // run FDTD in openmp with diamond tiling
    void update_FDTD_omp_dt(size_t num_timesteps);

    // run FDTD in gpu without diamond tiling 
    void update_FDTD_gpu(size_t num_timesteps);

    // check correctness
    bool check_correctness_gpu();
    bool check_correctness_omp();

  private:

    // get number of tiles
    // BLX is the length of the 1st row of a mountain, BLT is the number of time steps, Nx is space size
    size_t _get_num_of_tiles(size_t BLX, size_t BLT, size_t Nx);

    size_t _Nx;
    size_t _Ny;
    size_t _Nz;

    // E and H (initial field)
    std::vector<float> _Ex;
    std::vector<float> _Ey;
    std::vector<float> _Ez;
    std::vector<float> _Hx;
    std::vector<float> _Hy;
    std::vector<float> _Hz;

    // E and H (result from single thread CPU)
    std::vector<float> _Ex_seq;
    std::vector<float> _Ey_seq;
    std::vector<float> _Ez_seq;
    std::vector<float> _Hx_seq;
    std::vector<float> _Hy_seq;
    std::vector<float> _Hz_seq;

    // E and H (result from openmp)
    std::vector<float> _Ex_omp;
    std::vector<float> _Ey_omp;
    std::vector<float> _Ez_omp;
    std::vector<float> _Hx_omp;
    std::vector<float> _Hy_omp;
    std::vector<float> _Hz_omp;
    
    // E and H (result from GPU)
    std::vector<float> _Ex_gpu;
    std::vector<float> _Ey_gpu;
    std::vector<float> _Ez_gpu;
    std::vector<float> _Hx_gpu;
    std::vector<float> _Hy_gpu;
    std::vector<float> _Hz_gpu;

    // J and M (source)
    std::vector<float> _Jx;
    std::vector<float> _Jy;
    std::vector<float> _Jz;
    std::vector<float> _Mx;
    std::vector<float> _My;
    std::vector<float> _Mz;
    
    // Ca, Cb, Da, Db (coefficient)
    std::vector<float> _Cax;
    std::vector<float> _Cay;
    std::vector<float> _Caz;
    std::vector<float> _Cbx;
    std::vector<float> _Cby;
    std::vector<float> _Cbz;

    std::vector<float> _Dax;
    std::vector<float> _Day;
    std::vector<float> _Daz;
    std::vector<float> _Dbx;
    std::vector<float> _Dby;
    std::vector<float> _Dbz;

    // spacial sample rate
    float _dx = 1;
};  

void gDiamond::update_FDTD_seq(size_t num_timesteps) {

  // create temporary E and H for experiments
  std::vector<float> Ex_temp(_Nx * _Ny * _Nz);
  std::vector<float> Ey_temp(_Nx * _Ny * _Nz);
  std::vector<float> Ez_temp(_Nx * _Ny * _Nz);
  std::vector<float> Hx_temp(_Nx * _Ny * _Nz);
  std::vector<float> Hy_temp(_Nx * _Ny * _Nz);
  std::vector<float> Hz_temp(_Nx * _Ny * _Nz);
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    Ex_temp[i] = _Ex[i];
    Ey_temp[i] = _Ey[i];
    Ez_temp[i] = _Ez[i];
    Hx_temp[i] = _Hx[i];
    Hy_temp[i] = _Hy[i];
    Hz_temp[i] = _Hz[i];
  }

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(size_t k=1; k<_Nz-1; k++) {
      for(size_t j=1; j<_Ny-1; j++) {
        for(size_t i=1; i<_Nx-1; i++) {
          size_t idx = i + j*_Nx + k*(_Nx*_Ny);
          Ex_temp[idx] = _Cax[idx] * Ex_temp[idx] + _Cbx[idx] *
            ((Hz_temp[idx] - Hz_temp[idx - _Nx]) - (Hy_temp[idx] - Hy_temp[idx - _Nx * _Ny]) - _Jx[idx] * _dx);
          Ey_temp[idx] = _Cay[idx] * Ey_temp[idx] + _Cby[idx] *
            ((Hx_temp[idx] - Hx_temp[idx - _Nx * _Ny]) - (Hz_temp[idx] - Hz_temp[idx - 1]) - _Jy[idx] * _dx);
          Ez_temp[idx] = _Caz[idx] * Ez_temp[idx] + _Cbz[idx] *
            ((Hy_temp[idx] - Hy_temp[idx - 1]) - (Hx_temp[idx] - Hx_temp[idx - _Nx]) - _Jz[idx] * _dx);
        }
      }
    }

    // update H
    for(size_t k=1; k<_Nz-1; k++) {
      for(size_t j=1; j<_Ny-1; j++) {
        for(size_t i=1; i<_Nx-1; i++) {
          size_t idx = i + j*_Nx + k*(_Nx*_Ny);
          Hx_temp[idx] = _Dax[idx] * Hx_temp[idx] + _Dbx[idx] *
            ((Ey_temp[idx + _Nx * _Ny] - Ey_temp[idx]) - (Ez_temp[idx + _Nx] - Ez_temp[idx]) - _Mx[idx] * _dx);
          Hy_temp[idx] = _Day[idx] * Hy_temp[idx] + _Dby[idx] *
            ((Ez_temp[idx + 1] - Ez_temp[idx]) - (Ex_temp[idx + _Nx * _Ny] - Ex_temp[idx]) - _My[idx] * _dx);
          Hz_temp[idx] = _Daz[idx] * Hz_temp[idx] + _Dbz[idx] *
            ((Ex_temp[idx + _Nx] - Ex_temp[idx]) - (Ey_temp[idx + 1] - Ey_temp[idx]) - _Mz[idx] * _dx);
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "seq runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_seq[i] = Ex_temp[i];
    _Ey_seq[i] = Ey_temp[i];
    _Ez_seq[i] = Ez_temp[i];
    _Hx_seq[i] = Hx_temp[i];
    _Hy_seq[i] = Hy_temp[i];
    _Hz_seq[i] = Hz_temp[i];
  }
}

bool gDiamond::check_correctness_gpu() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_seq[i] - _Ex_gpu[i]) >= 1e-8 ||
       fabs(_Ey_seq[i] - _Ey_gpu[i]) >= 1e-8 ||
       fabs(_Ez_seq[i] - _Ez_gpu[i]) >= 1e-8 ||
       fabs(_Hx_seq[i] - _Hx_gpu[i]) >= 1e-8 ||
       fabs(_Hy_seq[i] - _Hy_gpu[i]) >= 1e-8 ||
       fabs(_Hz_seq[i] - _Hz_gpu[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
} 

bool gDiamond::check_correctness_omp() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_seq[i] - _Ex_omp[i]) >= 1e-8 ||
       fabs(_Ey_seq[i] - _Ey_omp[i]) >= 1e-8 ||
       fabs(_Ez_seq[i] - _Ez_omp[i]) >= 1e-8 ||
       fabs(_Hx_seq[i] - _Hx_omp[i]) >= 1e-8 ||
       fabs(_Hy_seq[i] - _Hy_omp[i]) >= 1e-8 ||
       fabs(_Hz_seq[i] - _Hz_omp[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
} 

size_t gDiamond::_get_num_of_tiles(size_t BLX, size_t BLT, size_t Nx) {

  // here we always set a mountain at the beginning of space
  
  size_t num_tiles;

  size_t mountain_bottom = BLX;
  size_t valley_top = BLX - 2*(BLT - 1) - 1;

  size_t two_tiles = mountain_bottom + valley_top;

  size_t num_two_tiles = Nx / two_tiles;
  size_t remain = Nx - num_two_tiles * two_tiles;
  size_t remain_tiles = 0;

  /*
        E E E |
      H H H H |
      E E E E | E
    H H H H H | H
    E E E E E | E E
  H H H H H H | H H
             check
  */
  size_t check = mountain_bottom - (BLT - 1);

  if(remain > 0 && remain <= check) {
    remain_tiles = 1;
  }
  else if(remain > check && remain < two_tiles) {
    remain_tiles = 2;
  }

  // + 1 since there is always a valley at the beginning
  num_tiles = 2*num_two_tiles + remain_tiles + 1;

  return num_tiles;
}

} // end of namespace gdiamond

#endif




























