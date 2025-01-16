#ifndef GDIAMOND_OMP_HPP
#define GDIAMOND_OMP_HPP

#include "gdiamond.hpp"
#include <omp.h>

namespace gdiamond {

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

  _get_indices_and_ranges(BLX, BLT, _Nx, _mountain_indices_X, _mountain_ranges_X, _valley_indices_X, _valley_ranges_X); 
  _get_indices_and_ranges(BLY, BLT, _Ny, _mountain_indices_Y, _mountain_ranges_Y, _valley_indices_Y, _valley_ranges_Y); 
  _get_indices_and_ranges(BLZ, BLT, _Nz, _mountain_indices_Z, _mountain_ranges_Z, _valley_indices_Z, _valley_ranges_Z); 

  // set ntiles for different phases
  for(size_t phase = 0; phase<max_phases; phase++) {
    std::vector<size_t> Entiles_onephase_X(BLT); 
    std::vector<size_t> Entiles_onephase_Y(BLT); 
    std::vector<size_t> Entiles_onephase_Z(BLT); 
    std::vector<size_t> Hntiles_onephase_X(BLT); 
    std::vector<size_t> Hntiles_onephase_Y(BLT); 
    std::vector<size_t> Hntiles_onephase_Z(BLT); 
    for(size_t t=0; t<BLT; t++) {
      size_t E_row = t*2;
      size_t H_row = t*2 + 1;
      std::vector<int> E_ntiles = _set_num_tiles(E_row, phase); 
      std::vector<int> H_ntiles = _set_num_tiles(H_row, phase); 
      Entiles_onephase_X[t] = E_ntiles[0];
      Entiles_onephase_Y[t] = E_ntiles[1];
      Entiles_onephase_Z[t] = E_ntiles[2];
      Hntiles_onephase_X[t] = H_ntiles[0];
      Hntiles_onephase_Y[t] = H_ntiles[1];
      Hntiles_onephase_Z[t] = H_ntiles[2];
    }
    _Entiles_phases_X.push_back(Entiles_onephase_X);
    _Entiles_phases_Y.push_back(Entiles_onephase_Y);
    _Entiles_phases_Z.push_back(Entiles_onephase_Z);
    _Hntiles_phases_X.push_back(Hntiles_onephase_X);
    _Hntiles_phases_Y.push_back(Hntiles_onephase_Y);
    _Hntiles_phases_Z.push_back(Hntiles_onephase_Z);
  }

  // get ranges for different rows and phases
  // update ntiles_phases again cuz some ranges got removed
  for(size_t phase = 0; phase<max_phases; phase++) {
    std::vector<std::vector<std::pair<size_t, size_t>>> Eranges_onephase_X; 
    std::vector<std::vector<std::pair<size_t, size_t>>> Eranges_onephase_Y; 
    std::vector<std::vector<std::pair<size_t, size_t>>> Eranges_onephase_Z; 
    std::vector<std::vector<std::pair<size_t, size_t>>> Hranges_onephase_X; 
    std::vector<std::vector<std::pair<size_t, size_t>>> Hranges_onephase_Y; 
    std::vector<std::vector<std::pair<size_t, size_t>>> Hranges_onephase_Z; 
    for(size_t t=0; t<BLT; t++) {
      std::vector<std::pair<size_t, size_t>> Eranges_onephase_X_onet; 
      std::vector<std::pair<size_t, size_t>> Eranges_onephase_Y_onet; 
      std::vector<std::pair<size_t, size_t>> Eranges_onephase_Z_onet; 
      std::vector<std::pair<size_t, size_t>> Hranges_onephase_X_onet; 
      std::vector<std::pair<size_t, size_t>> Hranges_onephase_Y_onet; 
      std::vector<std::pair<size_t, size_t>> Hranges_onephase_Z_onet; 
      size_t Entiles_onephase_X_size_adjust = 0; 
      size_t Entiles_onephase_Y_size_adjust = 0; 
      size_t Entiles_onephase_Z_size_adjust = 0; 
      size_t Hntiles_onephase_X_size_adjust = 0; 
      size_t Hntiles_onephase_Y_size_adjust = 0; 
      size_t Hntiles_onephase_Z_size_adjust = 0; 
      size_t E_row = t*2;
      size_t H_row = t*2 + 1;
      for(size_t xx=0; xx<_Entiles_phases_X[phase][t]; xx++) {
        std::vector<size_t> head_tail = _set_ranges_X(E_row, xx, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_X_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_X_size_adjust;
        }
      }
      _Entiles_phases_X[phase][t] -= Entiles_onephase_X_size_adjust;
      for(size_t yy=0; yy<_Entiles_phases_Y[phase][t]; yy++) {
        std::vector<size_t> head_tail = _set_ranges_Y(E_row, yy, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_Y_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_Y_size_adjust;
        }
      }
      _Entiles_phases_Y[phase][t] -= Entiles_onephase_Y_size_adjust;
      for(size_t zz=0; zz<_Entiles_phases_Z[phase][t]; zz++) {
        std::vector<size_t> head_tail = _set_ranges_Z(E_row, zz, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_Z_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_Z_size_adjust;
        }
      }
      _Entiles_phases_Z[phase][t] -= Entiles_onephase_Z_size_adjust;
      for(size_t xx=0; xx<_Hntiles_phases_X[phase][t]; xx++) {
        std::vector<size_t> head_tail = _set_ranges_X(H_row, xx, phase);
        if(head_tail[0] <= head_tail[1]) {
          Hranges_onephase_X_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Hntiles_onephase_X_size_adjust;
        }
      }
      _Hntiles_phases_X[phase][t] -= Hntiles_onephase_X_size_adjust;
      for(size_t yy=0; yy<_Hntiles_phases_Y[phase][t]; yy++) {
        std::vector<size_t> head_tail = _set_ranges_Y(H_row, yy, phase);
        if(head_tail[0] <= head_tail[1]) {
          Hranges_onephase_Y_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Hntiles_onephase_Y_size_adjust;
        }
      }
      _Hntiles_phases_Y[phase][t] -= Hntiles_onephase_Y_size_adjust;
      for(size_t zz=0; zz<_Hntiles_phases_Z[phase][t]; zz++) {
        std::vector<size_t> head_tail = _set_ranges_Z(H_row, zz, phase);
        if(head_tail[0] <= head_tail[1]) {
          Hranges_onephase_Z_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Hntiles_onephase_Z_size_adjust;
        }
      }
      _Hntiles_phases_Z[phase][t] -= Hntiles_onephase_Z_size_adjust;

      Eranges_onephase_X.push_back(Eranges_onephase_X_onet);
      Eranges_onephase_Y.push_back(Eranges_onephase_Y_onet);
      Eranges_onephase_Z.push_back(Eranges_onephase_Z_onet);
      Hranges_onephase_X.push_back(Hranges_onephase_X_onet);
      Hranges_onephase_Y.push_back(Hranges_onephase_Y_onet);
      Hranges_onephase_Z.push_back(Hranges_onephase_Z_onet);
    }
    _Eranges_phases_X.push_back(Eranges_onephase_X); 
    _Eranges_phases_Y.push_back(Eranges_onephase_Y); 
    _Eranges_phases_Z.push_back(Eranges_onephase_Z); 
    _Hranges_phases_X.push_back(Hranges_onephase_X); 
    _Hranges_phases_Y.push_back(Hranges_onephase_Y); 
    _Hranges_phases_Z.push_back(Hranges_onephase_Z); 
  }

  // std::cout << "_Entiles_phases_X = \n";
  // for(size_t phase = 0; phase<max_phases; phase++) {
  //   std::cout << "[";
  //   for(auto n : _Entiles_phases_X[phase]) {
  //     std::cout << n << " ";
  //   }
  //   std::cout << "]\n";
  // }

  // std::cout << "_Eranges_phases_X = \n";
  // for(size_t phase = 0; phase<max_phases; phase++) {
  //   std::cout << "[";
  //   for(auto n : _Eranges_phases_X[phase]) {
  //     std::cout << "(" << n.first << ", " << n.second << ") ";
  //   }
  //   std::cout << "]\n";
  // }


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

        #pragma omp parallel for collapse(2)
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

        #pragma omp parallel for collapse(2)
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
