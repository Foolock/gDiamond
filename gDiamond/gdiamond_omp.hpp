#ifndef GDIAMOND_OMP_HPP
#define GDIAMOND_OMP_HPP

#include "gdiamond.hpp"
#include <omp.h>

namespace gdiamond {

void gDiamond::update_FDTD_omp(size_t num_timesteps) {

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
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for collapse(2)
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
  std::cout << "omp runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_omp[i] = Ex_temp[i];
    _Ey_omp[i] = Ey_temp[i];
    _Ez_omp[i] = Ez_temp[i];
    _Hx_omp[i] = Hx_temp[i];
    _Hy_omp[i] = Hy_temp[i];
    _Hz_omp[i] = Hz_temp[i];
  }
}

void gDiamond::update_FDTD_omp_dt(size_t num_timesteps) {

  size_t BLX = 7;
  size_t BLT = 3;
  size_t Nx = 19;
  _get_indices_and_ranges(BLX, BLT, Nx); 

}

} // end of namespace gdiamond

#endif
