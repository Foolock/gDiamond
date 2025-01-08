#ifndef GDIAMOND_HPP 
#define GDIAMOND_HPP 

#include <iostream>
#include <random> 

namespace gdiamond {

class gDiamond {

  public:
    gDiamond(size_t Nx, size_t Ny, size_t Nz): _Nx(Nx), _Ny(Ny), _Nz(Nz),
                                               _Ex(Nx * Ny * Nz), _Ey(Nx * Ny * Nz), _Ez(Nx * Ny * Nz),
                                               _Hx(Nx * Ny * Nz), _Hy(Nx * Ny * Nz), _Hz(Nx * Ny * Nz),
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
      for(size_t i=0; i<Nx * Ny * Nz; i++) {
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
    update_FDTD_seq();

  private:
    size_t _Nx;
    size_t _Ny;
    size_t _Nz;

    // E and H
    std::vector<float> _Ex;
    std::vector<float> _Ey;
    std::vector<float> _Ez;
    std::vector<float> _Hx;
    std::vector<float> _Hy;
    std::vector<float> _Hz;

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

};  

gDiamond::update_FDTD_seq() {


}

} // end of namespace gdiamond

#endif




























