#ifndef GDIAMOND_OMP_HPP
#define GDIAMOND_OMP_HPP

#include "gdiamond.hpp"
#include <omp.h>

namespace gdiamond {

void gDiamond::update_FDTD_omp_figures(size_t num_timesteps) {

  if (std::filesystem::create_directory("omp_figures")) {
      std::cerr << "omp_figures created successfully. " << std::endl;
  } else {
      std::cerr << "failed to create omp_figures or it already exists." << std::endl;
      std::exit(EXIT_FAILURE);
  }

  // create temporary E and H for experiments
  std::vector<float> Ex_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ey_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ez_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hx_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hy_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hz_temp(_Nx * _Ny * _Nz, 0);

  // clear source Mz for experiments
  _Mz.clear();

  std::chrono::duration<double> omp_runtime(0);

  for(size_t t=0; t<num_timesteps; t++) {

    auto start = std::chrono::high_resolution_clock::now();
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;

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
    auto end = std::chrono::high_resolution_clock::now();
    omp_runtime += end-start;

    // Record the field using a monitor, once in a while
    if (t % (num_timesteps/10) == 0)
    {
      printf("Iter: %ld / %ld \n", t, num_timesteps);

      float *H_time_monitor_xy;
      H_time_monitor_xy = (float *)malloc(_Nx * _Ny * sizeof(float));
      memset(H_time_monitor_xy, 0, _Nx * _Ny * sizeof(float));

      // ------------ plotting time domain
      // File name initialization
      char field_filename[50];
      size_t k = _Nz / 2;  // Assuming you want the middle slice
      for(size_t i=0; i<_Nx; i++) {
        for(size_t j=0; j<_Ny; j++) {
          H_time_monitor_xy[i + j*_Nx] = Hz_temp[i + j*_Nz + k*_Nx*_Ny];
        }
      }

      snprintf(field_filename, sizeof(field_filename), "omp_figures/Hz_omp_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }
  }
  std::cout << "omp runtime (excluding figures output): " << omp_runtime.count() << "s\n"; 
  std::cout << "omp performance (excluding figures output): " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / omp_runtime.count() << "Mcells/s\n";

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_omp[i] = Ex_temp[i];
    _Ey_omp[i] = Ey_temp[i];
    _Ez_omp[i] = Ez_temp[i];
    _Hx_omp[i] = Hx_temp[i];
    _Hy_omp[i] = Hy_temp[i];
    _Hz_omp[i] = Hz_temp[i];
  }

}

void gDiamond::update_FDTD_omp(size_t num_timesteps) {

  // create temporary E and H for experiments
  std::vector<float> Ex_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ey_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ez_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hx_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hy_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hz_temp(_Nx * _Ny * _Nz, 0);

  // clear source Mz for experiments
  _Mz.clear();

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t t=0; t<num_timesteps; t++) {

    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;

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
  std::vector<float> Ex_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ey_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ez_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hx_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hy_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hz_temp(_Nx * _Ny * _Nz, 0);

  // let's first consider num_timesteps is a multiply of BLT
  size_t steps = num_timesteps / BLT;
  // apply diamond tilings to all X, Y, Z dimensions
  size_t max_phases = 8; 

  _setup_diamond_tiling(BLX, BLY, BLZ, BLT, max_phases);

  // clear source Mz for experiments
  _Mz.clear();

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t tt=0; tt<steps; tt++) {
    for(size_t phase=0; phase<max_phases; phase++) {
      for(size_t t=0; t<BLT; t+=1) { 

        // t + tt*steps is the global time step
        float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * (t + tt*BLT) * dt);
        // std::cout << "omp_dt: Mz_value = " << Mz_value << "\n";
        _Mz[_source_idx] = Mz_value;

        #pragma omp parallel for collapse(3) schedule(dynamic)
        for(size_t xx=0; xx<_Entiles_phases_X[phase][t]; xx++) {
          for(size_t yy=0; yy<_Entiles_phases_Y[phase][t]; yy++) {
            for(size_t zz=0; zz<_Entiles_phases_Z[phase][t]; zz++) {
              // update E
              for(size_t x=_Eranges_phases_X[phase][t][xx].first; x<=_Eranges_phases_X[phase][t][xx].second; x++) {
                for(size_t y=_Eranges_phases_Y[phase][t][yy].first; y<=_Eranges_phases_Y[phase][t][yy].second; y++) {
                  for(size_t z=_Eranges_phases_Z[phase][t][zz].first; z<=_Eranges_phases_Z[phase][t][zz].second; z++) {
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
            }
          }
        }

        #pragma omp parallel for collapse(3) schedule(dynamic)
        for(size_t xx=0; xx<_Hntiles_phases_X[phase][t]; xx++) {
          for(size_t yy=0; yy<_Hntiles_phases_Y[phase][t]; yy++) {
            for(size_t zz=0; zz<_Hntiles_phases_Z[phase][t]; zz++) {
              // update H
              for(size_t x=_Hranges_phases_X[phase][t][xx].first; x<=_Hranges_phases_X[phase][t][xx].second; x++) {
                for(size_t y=_Hranges_phases_Y[phase][t][yy].first; y<=_Hranges_phases_Y[phase][t][yy].second; y++) {
                  for(size_t z=_Hranges_phases_Z[phase][t][zz].first; z<=_Hranges_phases_Z[phase][t][zz].second; z++) {
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
