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

void gDiamond::update_FDTD_omp_dt(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t num_timesteps) {

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

  _get_indices_and_ranges(BLX, BLT, _Nx, _mountain_indices_X, _mountain_ranges_X, _valley_indices_X, _valley_ranges_X); 
  _get_indices_and_ranges(BLY, BLT, _Ny, _mountain_indices_Y, _mountain_ranges_Y, _valley_indices_Y, _valley_ranges_Y); 
  _get_indices_and_ranges(BLZ, BLT, _Nz, _mountain_indices_Z, _mountain_ranges_Z, _valley_indices_Z, _valley_ranges_Z); 

  auto start = std::chrono::high_resolution_clock::now();
  // let's first consider num_timesteps is a multiply of BLT
  size_t steps = num_timesteps / BLT;
  // apply diamond tilings to all X, Y, Z dimensions
  size_t max_phases = 8; 
  for(size_t tt=0; tt<steps; tt++) {
    for(size_t phase=0; phase<max_phases; phase++) {
      for(size_t t=0; t<BLT; t+=2) {
        #pragma omp parallel for collapse(2)
        for(size_t xx=0; xx<_mountain_indices_X[t].size(); xx++) {
          for(size_t yy=0; yy<_mountain_indices_Y[t].size(); yy++) {
            for(size_t zz=0; zz<_mountain_indices_Z[t].size(); zz++) {
              std::vector<int> results = _set_ranges(t, xx, yy, zz, phase);
              // update E
              for(size_t x=results[0]; x<=results[1]; x++) {
                for(size_t y=results[2]; y<=results[3]; y++) {
                  for(size_t z=results[4]; z<=results[5]; z++) {
                    size_t idx = x + y*_Nx + z*(_Nx*_Ny);
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
              for(size_t x=results[0]; x<=results[1]; x++) {
                for(size_t y=results[2]; y<=results[3]; y++) {
                  for(size_t z=results[4]; z<=results[5]; z++) {
                    size_t idx = x + y*_Nx + z*(_Nx*_Ny);
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
          }
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "omp_dt runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_omp_dt[i] = Ex_temp[i];
    _Ey_omp_dt[i] = Ey_temp[i];
    _Ez_omp_dt[i] = Ez_temp[i];
    _Hx_omp_dt[i] = Hx_temp[i];
    _Hy_omp_dt[i] = Hy_temp[i];
    _Hz_omp_dt[i] = Hz_temp[i];
  }
}

} // end of namespace gdiamond

#endif
