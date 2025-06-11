#ifndef GDIAMOND_HPP 
#define GDIAMOND_HPP 

#include <iostream>
#include <filesystem>
#include <random> 
#include <chrono>
#include <vector>
#include "utils.h"

namespace gdiamond {

struct Pt_idx {
  int x, y, z; 
};

class gDiamond {

  public:
    gDiamond(size_t Nx, size_t Ny, size_t Nz): _Nx(Nx), _Ny(Ny), _Nz(Nz),
                                               _Ex(Nx * Ny * Nz), _Ey(Nx * Ny * Nz), _Ez(Nx * Ny * Nz),
                                               _Hx(Nx * Ny * Nz), _Hy(Nx * Ny * Nz), _Hz(Nx * Ny * Nz),
                                               _Ex_seq(Nx * Ny * Nz), _Ey_seq(Nx * Ny * Nz), _Ez_seq(Nx * Ny * Nz),
                                               _Hx_seq(Nx * Ny * Nz), _Hy_seq(Nx * Ny * Nz), _Hz_seq(Nx * Ny * Nz),
                                               _Ex_gpu(Nx * Ny * Nz), _Ey_gpu(Nx * Ny * Nz), _Ez_gpu(Nx * Ny * Nz),
                                               _Hx_gpu(Nx * Ny * Nz), _Hy_gpu(Nx * Ny * Nz), _Hz_gpu(Nx * Ny * Nz),
                                               // _Ex_gpu_shEH(Nx * Ny * Nz), _Ey_gpu_shEH(Nx * Ny * Nz), _Ez_gpu_shEH(Nx * Ny * Nz),
                                               // _Hx_gpu_shEH(Nx * Ny * Nz), _Hy_gpu_shEH(Nx * Ny * Nz), _Hz_gpu_shEH(Nx * Ny * Nz),
                                               _Ex_gpu_bl(Nx * Ny * Nz), _Ey_gpu_bl(Nx * Ny * Nz), _Ez_gpu_bl(Nx * Ny * Nz),
                                               _Hx_gpu_bl(Nx * Ny * Nz), _Hy_gpu_bl(Nx * Ny * Nz), _Hz_gpu_bl(Nx * Ny * Nz),
                                               _Ex_simu(Nx * Ny * Nz), _Ey_simu(Nx * Ny * Nz), _Ez_simu(Nx * Ny * Nz),
                                               _Hx_simu(Nx * Ny * Nz), _Hy_simu(Nx * Ny * Nz), _Hz_simu(Nx * Ny * Nz),
                                               // _Ex_simu_init(Nx * Ny * Nz), _Ey_simu_init(Nx * Ny * Nz), _Ez_simu_init(Nx * Ny * Nz),
                                               // _Hx_simu_init(Nx * Ny * Nz), _Hy_simu_init(Nx * Ny * Nz), _Hz_simu_init(Nx * Ny * Nz),
                                               // _Ex_simu_sh(Nx * Ny * Nz), _Ey_simu_sh(Nx * Ny * Nz), _Ez_simu_sh(Nx * Ny * Nz),
                                               // _Hx_simu_sh(Nx * Ny * Nz), _Hy_simu_sh(Nx * Ny * Nz), _Hz_simu_sh(Nx * Ny * Nz),
                                               // _Ex_omp(Nx * Ny * Nz), _Ey_omp(Nx * Ny * Nz), _Ez_omp(Nx * Ny * Nz),
                                               // _Hx_omp(Nx * Ny * Nz), _Hy_omp(Nx * Ny * Nz), _Hz_omp(Nx * Ny * Nz),
                                               // _Ex_omp_dt(Nx * Ny * Nz), _Ey_omp_dt(Nx * Ny * Nz), _Ez_omp_dt(Nx * Ny * Nz),
                                               // _Hx_omp_dt(Nx * Ny * Nz), _Hy_omp_dt(Nx * Ny * Nz), _Hz_omp_dt(Nx * Ny * Nz),
                                               _Jx(Nx * Ny * Nz, 0), _Jy(Nx * Ny * Nz, 0), _Jz(Nx * Ny * Nz, 0),
                                               _Mx(Nx * Ny * Nz, 0), _My(Nx * Ny * Nz, 0), _Mz(Nx * Ny * Nz, 0),
                                               _Cax(Nx * Ny * Nz, 0), _Cay(Nx * Ny * Nz, 0), _Caz(Nx * Ny * Nz, 0),
                                               _Cbx(Nx * Ny * Nz, 0), _Cby(Nx * Ny * Nz, 0), _Cbz(Nx * Ny * Nz, 0),
                                               _Dax(Nx * Ny * Nz, 0), _Day(Nx * Ny * Nz, 0), _Daz(Nx * Ny * Nz, 0),
                                               _Dbx(Nx * Ny * Nz, 0), _Dby(Nx * Ny * Nz, 0), _Dbz(Nx * Ny * Nz, 0)
    {
      std::cerr << "initializing Ca, Cb, Da, Db...\n";

      _source_idx = Nx/2 + Ny/2 * Nx + Nz/2 * Nx * Ny;
    
      bool *mask;
      mask = (bool *)malloc(Nx * Ny * sizeof(bool));
      memset(mask, 0, Nx * Ny * sizeof(bool));
      Complex eps_air = Complex(1.0f, 0.0f);
      Complex eps_Si = Complex(12.0f, 0.001f);

      float t_slab = 200 * nm;  // slab thickness
      int t_slab_grid = std::round(t_slab / _dx);
      int k_mid = Nz / 2;
      int slab_k_min = k_mid - t_slab_grid / 2;
      int slab_k_max = slab_k_min + t_slab_grid;
      float h_PML = 1 * um;  // Thickness of PML, change according to prof. gu
      int t_PML = std::ceil(h_PML / _dx);
      set_FDTD_matrices_3D_structure(_Cax, _Cbx, _Cay, _Cby, _Caz, _Cbz,
                                     _Dax, _Dbx, _Day, _Dby, _Daz, _Dbz,
                                     Nx, Ny, Nz, _dx, dt, mask, eps_air, eps_Si,
                                     slab_k_min, slab_k_max, SOURCE_OMEGA, t_PML);


      std::cerr << "finish initialization\n";
      free(mask);
    }

    // for figures output
    void update_FDTD_seq_figures(size_t num_timesteps);
    void update_FDTD_omp_figures(size_t num_timesteps);
    void update_FDTD_gpu_figures(size_t num_timesteps); // only use for result checking

    // run FDTD in cpu single thread
    void update_FDTD_seq(size_t num_timesteps);
    void update_FDTD_seq_test(size_t num_timesteps);
    void update_FDTD_seq_check_result(size_t num_timesteps); // only use for result checking

    // run FDTD in openmp
    void update_FDTD_omp(size_t num_timesteps);

    // run FDTD in openmp with diamond tiling
    void update_FDTD_omp_dt(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t num_timesteps);

    // run FDTD in gpu  
    void update_FDTD_gpu_2D(size_t num_timesteps); // 2-D mapping, without diamond tiling, no pipeline
    void update_FDTD_gpu_3D_warp_underutilization(size_t num_timesteps); // 3-D mapping, has warp underutilization issue 
    void update_FDTD_gpu_3D_warp_underutilization_fix(size_t num_timesteps); // 3-D mapping, fix warp underutilization issue 
    /*
      this implementation assume X, Y, Z dimension is a multiple of 4
    */
    void update_FDTD_gpu_3D_warp_underutilization_fix_vectorized(size_t num_timesteps); // 3-D mapping, fix warp underutilization issue 
    void update_FDTD_gpu_check_result(size_t num_timesteps); // only use for result checking

    // wrong implementation
    void update_FDTD_gpu_fuse_kernel(size_t num_timesteps); // 3-D mapping, using diamond tiling to fuse kernels
    void update_FDTD_gpu_fuse_kernel_testing(size_t num_timesteps); // 3-D mapping, using diamond tiling to fuse kernels

    // simulation check
    void update_FDTD_gpu_simulation(size_t num_timesteps); // simulation of gpu threads
    void update_FDTD_gpu_simulation_shmem_EH(size_t num_timesteps); // simulation of gpu threads, with shared memory on E, H
    void update_FDTD_gpu_simulation_check(size_t num_timesteps);
    void update_FDTD_gpu_simulation_1_D(size_t num_timesteps); // CPU single thread 1-D simulation of GPU workflow, diamond tiling 
    void update_FDTD_gpu_simulation_1_D_shmem(size_t num_timesteps); // CPU single thread 1-D simulation of GPU workflow, diamond tiling
    void update_FDTD_gpu_simulation_1_D_pt(size_t num_timesteps); // CPU single thread 1-D simulation of GPU workflow, parallelogram tiling 
    void update_FDTD_gpu_simulation_1_D_pt_shmem(size_t num_timesteps); // CPU single thread 1-D simulation of GPU workflow, parallelogram tiling 
    void update_FDTD_gpu_simulation_2_D_globalmem(size_t num_timesteps); // 2-D mapping, each thread finish the entire Z dimension,
    void update_FDTD_gpu_simulation_2_D_shmem(size_t num_timesteps); // 2-D mapping, each thread finish the entire Z dimension,
    void update_FDTD_gpu_simulation_1_D_pt_pipeline(size_t num_timesteps); // CPU single thread simulation of GPU workflow, parallelogram tiling with pipeline
    void update_FDTD_gpu_simulation_1_D_mil(size_t num_timesteps); // CPU single thread 1-D simulation of GPU workflow, more is less tiling
    void update_FDTD_gpu_simulation_3_D_mil(size_t num_timesteps); // CPU single thread 3-D simulation of GPU workflow, more is less tiling
    void update_FDTD_gpu_simulation_1_D_pt_pl(size_t num_tiemsteps); // CPU single thread 1-D simulation of GPU workflow, parallelogram tiling, pipeline

    // correct implementation after simulation
    void update_FDTD_gpu_fuse_kernel_globalmem(size_t num_timesteps); // 3-D mapping, using diamond tiling to fuse kernels, global memory only
    void update_FDTD_gpu_fuse_kernel_globalmem_pt(size_t num_timesteps); // 2-D mapping, using diamond tiling on X, Y dimension to fuse kernels,  
    void update_FDTD_gpu_fuse_kernel_shmem_EH(size_t num_timesteps); // 3-D mapping, using diamond tiling to fuse kernels, put EH in shared memory 
    void update_FDTD_gpu_fuse_kernel_shmem_EH_pt(size_t num_timesteps); // 2-D mapping, using diamond tiling on X, Y dimension to fuse kernels, put EH in shared memory 
                                                                        // for shared memory flushing, no optimization used, all loaded from globalmem directly
    void update_FDTD_gpu_fuse_kernel_shmem_EH_mil(size_t num_timesteps); // 3-D mapping, using more is less tiling on X, Y, Z dimension to fuse kernels, put EH in shared memory 

    // my parallelogram tiling idea
    void update_FDTD_cpu_simulation_1_D_pt(size_t num_timesteps); // CPU single thread 1-D simulation of parallelogram tiling
    void update_FDTD_cpu_simulation_3_D_pt(size_t num_timesteps); // CPU single thread 3-D simulation of parallelogram tiling
    void update_FDTD_gpu_pt(size_t num_timesteps); // GPU 3-D implementation of parallelogram tiling

    // fuse equations
    void update_FDTD_gpu_fq(size_t num_timesteps); // GPU fuse equation try 

    // check shared memory without limitation of dependencies, upper bound speedup check
    void update_FDTD_gpu_shmem_no_deps_obeyed(size_t num_timesteps); // GPU 3-D implementation naive, with shared memory, but does not obey dependencies

    // reimplemented diamond tiling based on proper global memory access pattern
    void update_FDTD_cpu_simulation_dt_1_D(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz); // CPU single thread 1-D simulation of diamond tiling, reimplemented
    void update_FDTD_cpu_simulation_dt_3_D(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz); // CPU single thread 3-D simulation of diamond tiling, reimplemented
                                                                                                   // 2-D mapping used
    void update_FDTD_cpu_simulation_dt_1_D_sdf(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz); // CPU single thread 1-D simulation of diamond tiling, reimplemented
                                                                                                       // sdf stands for src, dst, final array
    void update_FDTD_cpu_simulation_dt_3_D_sdf(size_t num_timesteps, size_t Tx); // CPU single thread 3-D simulation of diamond tiling, reimplemented
                                                                                                       // sdf stands for src, dst, final array
    void update_FDTD_cpu_simulation_dt_1_D_extra_copy(size_t num_timesteps, size_t Tx); // CPU single thread 1-D simulation of diamond tiling, reimplemented
                                                                                        // with extra copy to dst array in phase 1 and phase 2
    void update_FDTD_cpu_simulation_dt_3_D_extra_copy(size_t num_timesteps, size_t Tx); // CPU single thread 3-D simulation of diamond tiling, reimplemented
                                                                                        // with extra copy to dst in phase 1 and phase 2, no final needed

    // mix mapping for 3D diamond tiling
    // ver 2 is clean diamond tiling (verified)
    // ver 3 is replication tiling (verified) 
    // ver4 is wrong... ver5 has not been completed due to shared memory problem (increased complexity) 
    void update_FDTD_mix_mapping_sequential(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz); // simulate GPU workflow  
    void update_FDTD_mix_mapping_sequential_ver2(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_sequential_ver3(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_sequential_ver4(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_sequential_ver5(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_gpu(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_gpu_ver2(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_gpu_ver3(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   
    void update_FDTD_mix_mapping_gpu_ver4(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);   

    // replication tiling with larger shmem
    void update_FDTD_replication_tiling_sequential(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz);

    // check correctness
    bool check_correctness_gpu_2D();
    bool check_correctness_gpu();
    bool check_correctness_gpu_shmem();
    bool check_correctness_omp();
    bool check_correctness_omp_dt();
    bool check_correctness_simu();
    bool check_correctness_simu_shmem();
    bool check_correctness_rep();
    
    void print_results();

  private:

    // mix mapping for 3D diamond tiling
    template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
    void _updateEH_mix_mapping(std::vector<float>& Ex_pad, std::vector<float>& Ey_pad, std::vector<float>& Ez_pad,
                               std::vector<float>& Hx_pad, std::vector<float>& Hy_pad, std::vector<float>& Hz_pad,
                               const std::vector<float>& Cax, const std::vector<float>& Cbx,
                               const std::vector<float>& Cay, const std::vector<float>& Cby,
                               const std::vector<float>& Caz, const std::vector<float>& Cbz,
                               const std::vector<float>& Dax, const std::vector<float>& Dbx,
                               const std::vector<float>& Day, const std::vector<float>& Dby,
                               const std::vector<float>& Daz, const std::vector<float>& Dbz,
                               const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                               const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int Nx_pad, int Ny_pad, int Nz_pad, 
                               int xx_num, int yy_num, int zz_num, 
                               const std::vector<int>& xx_heads, 
                               const std::vector<int>& yy_heads,
                               const std::vector<int>& zz_heads,
                               size_t block_size,
                               size_t grid_size);

    template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
    void _updateEH_mix_mapping_ver2(std::vector<float>& Ex_pad, std::vector<float>& Ey_pad, std::vector<float>& Ez_pad,
                                    std::vector<float>& Hx_pad, std::vector<float>& Hy_pad, std::vector<float>& Hz_pad,
                                    const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                    const std::vector<float>& Cay, const std::vector<float>& Cby,
                                    const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                    const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                    const std::vector<float>& Day, const std::vector<float>& Dby,
                                    const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                    const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                    const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                    float dx, 
                                    int Nx, int Ny, int Nz,
                                    int Nx_pad, int Ny_pad, int Nz_pad, 
                                    int xx_num, int yy_num, int zz_num, 
                                    const std::vector<int>& xx_heads, 
                                    const std::vector<int>& yy_heads,
                                    const std::vector<int>& zz_heads,
                                    size_t block_size,
                                    size_t grid_size);

    // notice that for overlapped mix mapping ver3, we need to have a src and dst copy of E, H data
    void _updateEH_mix_mapping_ver3(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                    std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                    std::vector<float>& Ex_pad_dst, std::vector<float>& Ey_pad_dst, std::vector<float>& Ez_pad_dst,
                                    std::vector<float>& Hx_pad_dst, std::vector<float>& Hy_pad_dst, std::vector<float>& Hz_pad_dst,
                                    const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                    const std::vector<float>& Cay, const std::vector<float>& Cby,
                                    const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                    const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                    const std::vector<float>& Day, const std::vector<float>& Dby,
                                    const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                    const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                    const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                    float dx, 
                                    int Nx, int Ny, int Nz,
                                    int Nx_pad, int Ny_pad, int Nz_pad, 
                                    int xx_num, int yy_num, int zz_num, 
                                    const std::vector<int>& xx_heads, 
                                    const std::vector<int>& yy_heads,
                                    const std::vector<int>& zz_heads,
                                    size_t block_size,
                                    size_t grid_size);

    // notice that for overlapped mix mapping ver4, we need to have a src and dst copy of E, H data
    // we also need one more copy to store the replication result
    // that copy does not need to have full length, but we will optimize that later
    void _updateEH_mix_mapping_ver4(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                    std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                    std::vector<float>& Ex_pad_rep, std::vector<float>& Ey_pad_rep, std::vector<float>& Ez_pad_rep,
                                    std::vector<float>& Hx_pad_rep, std::vector<float>& Hy_pad_rep, std::vector<float>& Hz_pad_rep,
                                    std::vector<float>& Ex_pad_dst, std::vector<float>& Ey_pad_dst, std::vector<float>& Ez_pad_dst,
                                    std::vector<float>& Hx_pad_dst, std::vector<float>& Hy_pad_dst, std::vector<float>& Hz_pad_dst,
                                    const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                    const std::vector<float>& Cay, const std::vector<float>& Cby,
                                    const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                    const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                    const std::vector<float>& Day, const std::vector<float>& Dby,
                                    const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                    const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                    const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                    float dx, 
                                    int Nx, int Ny, int Nz,
                                    int Nx_pad, int Ny_pad, int Nz_pad, 
                                    int xx_num, int yy_num, int zz_num, 
                                    const std::vector<int>& xx_heads, 
                                    const std::vector<int>& yy_heads,
                                    const std::vector<int>& zz_heads,
                                    size_t block_size,
                                    size_t grid_size);

    void _updateEH_mix_mapping_ver5(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                    std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                    std::vector<float>& Ex_pad_temp, std::vector<float>& Ey_pad_temp, std::vector<float>& Ez_pad_temp,
                                    std::vector<float>& Hx_pad_temp, std::vector<float>& Hy_pad_temp, std::vector<float>& Hz_pad_temp,
                                    const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                    const std::vector<float>& Cay, const std::vector<float>& Cby,
                                    const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                    const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                    const std::vector<float>& Day, const std::vector<float>& Dby,
                                    const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                    const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                    const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                    float dx, 
                                    int Nx, int Ny, int Nz,
                                    int Nx_pad, int Ny_pad, int Nz_pad, 
                                    int xx_num, int yy_num, int zz_num, 
                                    const std::vector<int>& xx_heads, 
                                    const std::vector<int>& yy_heads,
                                    const std::vector<int>& zz_heads,
                                    int num_subtiles,
                                    int hyperplane_head,
                                    const std::vector<Pt_idx>& hyperplanes,
                                    size_t block_size,
                                    size_t grid_size);

    // for diamond tiling reimplementation
    void _updateEH_dt_1D_mountain_seq_extra_copy(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                      const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                      std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                                      std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                                      const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                      const std::vector<float>& Cay, const std::vector<float>& Cby,
                                      const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                      const std::vector<float>& Day, const std::vector<float>& Dby,
                                      const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int Nx_pad, 
                                      int xx_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads_mountain, 
                                      std::vector<int> xx_heads_valley, 
                                      size_t block_size,
                                      size_t grid_size, 
                                      size_t tt); 

  void _updateEH_dt_1D_valley_seq_extra_copy(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                      const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                      std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                                      std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                                      const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                      const std::vector<float>& Cay, const std::vector<float>& Cby,
                                      const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                      const std::vector<float>& Day, const std::vector<float>& Dby,
                                      const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int Nx_pad, 
                                      int xx_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads_mountain, 
                                      std::vector<int> xx_heads_valley, 
                                      size_t block_size,
                                      size_t grid_size, 
                                      size_t tt); 

    void _extract_original_from_padded_1D(const std::vector<float>& padded,
                                       std::vector<float>& origin,
                                       int Nx, int Ny, int Nz,
                                       int Nx_pad, 
                                       int left_pad) {

      for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
          for (int x = 0; x < Nx; ++x) {
            int origin_idx = x + Nx * (y + Ny * z);
            int x_out = x + left_pad;
            int padded_idx = x_out + Nx_pad * (y + Ny * z);
            origin[origin_idx] = padded[padded_idx];
          }
        }
      }
    }


    void _updateEH_dt_1D_mountain_seq(std::vector<float>& Ex_src, std::vector<float>& Ey_src, std::vector<float>& Ez_src,
                                      std::vector<float>& Hx_src, std::vector<float>& Hy_src, std::vector<float>& Hz_src,
                                      std::vector<float>& Ex_final, std::vector<float>& Ey_final, std::vector<float>& Ez_final,
                                      std::vector<float>& Hx_final, std::vector<float>& Hy_final, std::vector<float>& Hz_final,
                                      const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                      const std::vector<float>& Cay, const std::vector<float>& Cby,
                                      const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                      const std::vector<float>& Day, const std::vector<float>& Dby,
                                      const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int Nx_pad, 
                                      int xx_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads, 
                                      size_t block_size,
                                      size_t grid_size, 
                                      size_t tt); 

    void _updateEH_dt_1D_valley_seq(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                    const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                    std::vector<float>& Ex_final, std::vector<float>& Ey_final, std::vector<float>& Ez_final,
                                    std::vector<float>& Hx_final, std::vector<float>& Hy_final, std::vector<float>& Hz_final,
                                    const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                    const std::vector<float>& Cay, const std::vector<float>& Cby,
                                    const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                    const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                    const std::vector<float>& Day, const std::vector<float>& Dby,
                                    const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                    const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                    const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                    float dx, 
                                    int Nx, int Ny, int Nz,
                                    int Nx_pad, 
                                    int xx_num, // number of tiles in each dimensions
                                    std::vector<int> xx_heads, 
                                    size_t block_size,
                                    size_t grid_size,
                                    size_t tt); 

    void padXY_1D_col_major(
      const std::vector<float>& input,
      std::vector<float>& output,
      int Nx, int Ny, int Nz,
      int left_pad,
      int right_pad) 
    {
      int Nx_pad = left_pad + Nx + right_pad;
      int Ny_pad = left_pad + Ny + right_pad;

      for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
          for (int x = 0; x < Nx; ++x) {
            int input_idx = x + y * Nx + z * Nx * Ny;
            int out_x = x + left_pad;
            int out_y = y + left_pad;
            int output_idx = out_x + out_y * Nx_pad + z * Nx_pad * Ny_pad;
            output[output_idx] = input[input_idx];
          }
        }
      }
    }

    void _extract_original_from_padded(const std::vector<float>& padded,
                                       std::vector<float>& origin,
                                       int Nx, int Ny, int Nz,
                                       int Nx_pad, int Ny_pad,
                                       int left_pad) {

      for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
          for (int x = 0; x < Nx; ++x) {
            int origin_idx = x + Nx * (y + Ny * z);
            int x_out = x + left_pad;
            int y_out = y + left_pad;
            int padded_idx = x_out + Nx_pad * (y_out + Ny_pad * z);
            origin[origin_idx] = padded[padded_idx];
          }
        }
      }
    }

    void _updateEH_dt_2D_seq(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                             const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                             std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                             std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                             const std::vector<float>& Cax, const std::vector<float>& Cbx,
                             const std::vector<float>& Cay, const std::vector<float>& Cby,
                             const std::vector<float>& Caz, const std::vector<float>& Cbz,
                             const std::vector<float>& Dax, const std::vector<float>& Dbx,
                             const std::vector<float>& Day, const std::vector<float>& Dby,
                             const std::vector<float>& Daz, const std::vector<float>& Dbz,
                             const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                             const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                             float dx, 
                             int Nx, int Ny, int Nz,
                             int Nx_pad, int Ny_pad,
                             int xx_num, int yy_num, // number of tiles in each dimensions
                             bool is_mountain_X, bool is_mountain_Y,
                             std::vector<int> xx_heads, 
                             std::vector<int> yy_heads, 
                             int left_pad,
                             size_t block_size,
                             size_t grid_size); 
    
    // for parallelogram tiling CPU simulation
    void _find_diagonal_hyperplanes(int Tx, int Ty, int Tz, 
                                    std::vector<Pt_idx>& hyperplanes, 
                                    std::vector<int>& hyperplane_heads, 
                                    std::vector<int>& hyperplanes_sizes) {
      int total_pixels = 0;
      int useful_tiles = 0;

      // Generate hyperplanes and store them in a flattened array
      for (int d = 0; d < Tx + Ty + Tz - 2; ++d) {
        hyperplane_heads.push_back(total_pixels); // Start index of current hyperplane
        int count = 0;
        for (int x = 0; x < Tx; ++x) {
          for (int y = 0; y < Ty; ++y) {
            for (int z = 0; z < Tz; ++z) {
              if (x + y + z == d) {
                hyperplanes.push_back({x, y, z});
                count++;
                if(x != 0 && y != 0 && z != 0) {
                  useful_tiles++;
                }
              }
            }
          }
        }
        hyperplanes_sizes.push_back(count); // Store number of pixels in this hyperplane
        total_pixels += count;
      }

      // Print all hyperplanes
      for (size_t d = 0; d < hyperplane_heads.size(); ++d) {
        std::cout << "Hyperplane " << d << ": ";
        int startIdx = hyperplane_heads[d];
        int size = hyperplanes_sizes[d];
        for (int i = 0; i < size; ++i) {
          Pt_idx p = hyperplanes[startIdx + i];
          std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
        }
        std::cout << std::endl;
      }

      std::cout << "number of useful tiles = " << useful_tiles << ", useful tile % = " << (useful_tiles * 1.0) / ((Tx * Ty * Tz) * 1.0) << "\n";

    }

    void _updateEH_pt_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                          std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                          std::vector<float>& Cax, std::vector<float>& Cbx,
                          std::vector<float>& Cay, std::vector<float>& Cby,
                          std::vector<float>& Caz, std::vector<float>& Cbz,
                          std::vector<float>& Dax, std::vector<float>& Dbx,
                          std::vector<float>& Day, std::vector<float>& Dby,
                          std::vector<float>& Daz, std::vector<float>& Dbz,
                          std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                          std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                          float dx, 
                          int Nx, int Ny, int Nz,
                          int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                          std::vector<int> xx_heads, 
                          std::vector<int> yy_heads, 
                          std::vector<int> zz_heads,
                          const std::vector<Pt_idx>& hyperplanes,
                          int hyperplane_head,
                          size_t block_size,
                          size_t grid_size); 

    // for update_FDTD_gpu_simulation
    void _updateEH_phase_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                             std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                             std::vector<float>& Cax, std::vector<float>& Cbx,
                             std::vector<float>& Cay, std::vector<float>& Cby,
                             std::vector<float>& Caz, std::vector<float>& Cbz,
                             std::vector<float>& Dax, std::vector<float>& Dbx,
                             std::vector<float>& Day, std::vector<float>& Dby,
                             std::vector<float>& Daz, std::vector<float>& Dbz,
                             std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                             std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                             float dx, 
                             int Nx, int Ny, int Nz,
                             int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                             std::vector<int> xx_heads, 
                             std::vector<int> yy_heads, 
                             std::vector<int> zz_heads,
                             std::vector<int> xx_tails, 
                             std::vector<int> yy_tails, 
                             std::vector<int> zz_tails,
                             int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                             size_t block_size,
                             size_t grid_size);

    void _updateEH_phase_seq_2D(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                std::vector<float>& Cax, std::vector<float>& Cbx,
                                std::vector<float>& Cay, std::vector<float>& Cby,
                                std::vector<float>& Caz, std::vector<float>& Cbz,
                                std::vector<float>& Dax, std::vector<float>& Dbx,
                                std::vector<float>& Day, std::vector<float>& Dby,
                                std::vector<float>& Daz, std::vector<float>& Dbz,
                                std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                float dx, 
                                int Nx, int Ny, int Nz,
                                int xx_num, int yy_num, 
                                std::vector<int> xx_heads, 
                                std::vector<int> yy_heads, 
                                std::vector<int> xx_tails, 
                                std::vector<int> yy_tails, 
                                int m_or_v_X, int m_or_v_Y, 
                                size_t block_size,
                                size_t grid_size);

    void _updateEH_phase_seq_2D_shmem_EH(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                         std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                         std::vector<float>& Cax, std::vector<float>& Cbx,
                                         std::vector<float>& Cay, std::vector<float>& Cby,
                                         std::vector<float>& Caz, std::vector<float>& Cbz,
                                         std::vector<float>& Dax, std::vector<float>& Dbx,
                                         std::vector<float>& Day, std::vector<float>& Dby,
                                         std::vector<float>& Daz, std::vector<float>& Dbz,
                                         std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                         std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                         float dx, 
                                         int Nx, int Ny, int Nz,
                                         int xx_num, int yy_num, 
                                         std::vector<int> xx_heads, 
                                         std::vector<int> yy_heads, 
                                         std::vector<int> xx_tails, 
                                         std::vector<int> yy_tails, 
                                         int m_or_v_X, int m_or_v_Y, 
                                         size_t block_size,
                                         size_t grid_size);

    void _updateEH_phase_seq_shmem_EH(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                      std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                      std::vector<float>& Cax, std::vector<float>& Cbx,
                                      std::vector<float>& Cay, std::vector<float>& Cby,
                                      std::vector<float>& Caz, std::vector<float>& Cbz,
                                      std::vector<float>& Dax, std::vector<float>& Dbx,
                                      std::vector<float>& Day, std::vector<float>& Dby,
                                      std::vector<float>& Daz, std::vector<float>& Dbz,
                                      std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                      std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads, 
                                      std::vector<int> yy_heads, 
                                      std::vector<int> zz_heads,
                                      std::vector<int> xx_tails, 
                                      std::vector<int> yy_tails, 
                                      std::vector<int> zz_tails,
                                      int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                                      size_t &total_cal,
                                      size_t current_time,
                                      size_t block_size,
                                      size_t grid_size);

    void _updateEH_phase_E_only_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                               std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                               std::vector<float>& Cax, std::vector<float>& Cbx,
                               std::vector<float>& Cay, std::vector<float>& Cby,
                               std::vector<float>& Caz, std::vector<float>& Cbz,
                               std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               std::vector<int> xx_heads, 
                               std::vector<int> yy_heads, 
                               std::vector<int> zz_heads,
                               std::vector<int> xx_tails, 
                               std::vector<int> yy_tails, 
                               std::vector<int> zz_tails,
                               size_t block_size,
                               size_t grid_size
                               ); 

    void _updateEH_phase_H_only_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                               std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                               std::vector<float>& Dax, std::vector<float>& Dbx,
                               std::vector<float>& Day, std::vector<float>& Dby,
                               std::vector<float>& Daz, std::vector<float>& Dbz,
                               std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               std::vector<int> xx_heads, 
                               std::vector<int> yy_heads, 
                               std::vector<int> zz_heads,
                               std::vector<int> xx_tails, 
                               std::vector<int> yy_tails, 
                               std::vector<int> zz_tails,
                               size_t block_size,
                               size_t grid_size
                               ); 

    void _updateEH_mil_seq(std::vector<float>& Ex_init, std::vector<float>& Ey_init, std::vector<float>& Ez_init,
                           std::vector<float>& Hx_init, std::vector<float>& Hy_init, std::vector<float>& Hz_init,
                           std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                           std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                           std::vector<float>& Cax, std::vector<float>& Cbx,
                           std::vector<float>& Cay, std::vector<float>& Cby,
                           std::vector<float>& Caz, std::vector<float>& Cbz,
                           std::vector<float>& Dax, std::vector<float>& Dbx,
                           std::vector<float>& Day, std::vector<float>& Dby,
                           std::vector<float>& Daz, std::vector<float>& Dbz,
                           std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                           std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                           float dx, 
                           int Nx, int Ny, int Nz,
                           int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                           std::vector<int> xx_heads, std::vector<int> yy_heads, std::vector<int> zz_heads,
                           std::vector<int> xx_tails, std::vector<int> yy_tails, std::vector<int> zz_tails,
                           std::vector<int> xx_top_heads, std::vector<int> yy_top_heads, std::vector<int> zz_top_heads,
                           std::vector<int> xx_top_tails, std::vector<int> yy_top_tails, std::vector<int> zz_top_tails,
                           size_t block_size,
                           size_t grid_size);

    // return {Ehead, Etail, Hhead, Htail}
    std::vector<int> _get_head_tail(size_t BLX, size_t BLT,
                                    std::vector<int> xx_heads, std::vector<int> xx_tails,
                                    size_t xx, size_t t,
                                    int mountain_or_valley,
                                    int Nx,
                                    int *calculate_E, int *calculate_H);

    int _get_z_planeE(int t, int zz, int Nz);
    int _get_z_planeH(int t, int zz, int Nz);
    int _get_z_planeE_shmem(int t, int zz, int Nz);
    int _get_z_planeH_shmem(int t, int zz, int Nz);

    // fill up indices and ranges vector for mountains and valleys
    void _get_indices_and_ranges(size_t BLX, size_t BLT, size_t Nx,
                                 std::vector<std::vector<size_t>>& mountain_indices,
                                 std::vector<std::vector<size_t>>& mountain_ranges,
                                 std::vector<std::vector<size_t>>& valley_indices,
                                 std::vector<std::vector<size_t>>& valley_ranges
                                 );

    // fill up indices and ranges for diamond_tiling
    void _setup_diamond_tiling(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t max_phases);
    void _setup_diamond_tiling_gpu(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t max_phases);

    // set tile ranges according to phases (the range of a specific mountain/valley)
    std::vector<int> _set_ranges(size_t row, size_t xx, size_t yy, size_t zz, size_t phase);
    std::vector<size_t> _set_ranges_X(size_t row, size_t xx, size_t phase);
    std::vector<size_t> _set_ranges_Y(size_t row, size_t yy, size_t phase);
    std::vector<size_t> _set_ranges_Z(size_t row, size_t zz, size_t phase);
    std::vector<size_t> _set_ranges_X_gpu(size_t row, size_t xx, size_t phase);
    std::vector<size_t> _set_ranges_Y_gpu(size_t row, size_t yy, size_t phase);
    std::vector<size_t> _set_ranges_Z_gpu(size_t row, size_t zz, size_t phase);

    // set number of tiles according to phases (the number of mountains/valleys)
    std::vector<int> _set_num_tiles(size_t row, size_t phase);

    // diamond tiles
    // mountain indices in different time steps
    std::vector<std::vector<size_t>> _mountain_indices_X;
    std::vector<std::vector<size_t>> _mountain_indices_Y;
    std::vector<std::vector<size_t>> _mountain_indices_Z;
    // mountain range in different time steps
    // e.g., the index range of 1st mountain in the 1st time step is 
    // [mountain_indices[0], mountain_indices[0][0] + mountain_ranges[0][0] - 1] 
    std::vector<std::vector<size_t>> _mountain_ranges_X;
    std::vector<std::vector<size_t>> _mountain_ranges_Y;
    std::vector<std::vector<size_t>> _mountain_ranges_Z;

    // valley indices
    std::vector<std::vector<size_t>> _valley_indices_X;
    std::vector<std::vector<size_t>> _valley_indices_Y;
    std::vector<std::vector<size_t>> _valley_indices_Z;
    // valley (bottom) range
    std::vector<std::vector<size_t>> _valley_ranges_X;
    std::vector<std::vector<size_t>> _valley_ranges_Y;
    std::vector<std::vector<size_t>> _valley_ranges_Z;

    // ntiles for 8 phases
    // 1st dimension is BLT
    // 2nd dimension is phases
    std::vector<std::vector<size_t>> _Entiles_phases_X;
    std::vector<std::vector<size_t>> _Entiles_phases_Y;
    std::vector<std::vector<size_t>> _Entiles_phases_Z;
    std::vector<std::vector<size_t>> _Hntiles_phases_X;
    std::vector<std::vector<size_t>> _Hntiles_phases_Y;
    std::vector<std::vector<size_t>> _Hntiles_phases_Z;
    
    // ranges for 8 phases
    // 1st dimension is xx (which mountains), described by pair<head, tail>
    // 2nd dimension is BLT 
    // 3rd dimension is phases
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Eranges_phases_X;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Eranges_phases_Y;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Eranges_phases_Z;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Hranges_phases_X;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Hranges_phases_Y;
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> _Hranges_phases_Z;

    size_t _Nx;
    size_t _Ny;
    size_t _Nz;

    size_t _source_idx;

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

    // E and H (result from openmp and diamond tiling)
    std::vector<float> _Ex_omp_dt;
    std::vector<float> _Ey_omp_dt;
    std::vector<float> _Ez_omp_dt;
    std::vector<float> _Hx_omp_dt;
    std::vector<float> _Hy_omp_dt;
    std::vector<float> _Hz_omp_dt;
    
    // E and H (result from GPU)
    std::vector<float> _Ex_gpu_bl;
    std::vector<float> _Ey_gpu_bl;
    std::vector<float> _Ez_gpu_bl;
    std::vector<float> _Hx_gpu_bl;
    std::vector<float> _Hy_gpu_bl;
    std::vector<float> _Hz_gpu_bl;
    std::vector<float> _Ex_gpu;
    std::vector<float> _Ey_gpu;
    std::vector<float> _Ez_gpu;
    std::vector<float> _Hx_gpu;
    std::vector<float> _Hy_gpu;
    std::vector<float> _Hz_gpu;
    std::vector<float> _Ex_gpu_shEH;
    std::vector<float> _Ey_gpu_shEH;
    std::vector<float> _Ez_gpu_shEH;
    std::vector<float> _Hx_gpu_shEH;
    std::vector<float> _Hy_gpu_shEH;
    std::vector<float> _Hz_gpu_shEH;

    // E and H (result from seq GPU simulation)
    std::vector<float> _Ex_simu;
    std::vector<float> _Ey_simu;
    std::vector<float> _Ez_simu;
    std::vector<float> _Hx_simu;
    std::vector<float> _Hy_simu;
    std::vector<float> _Hz_simu;
    std::vector<float> _Ex_simu_init;
    std::vector<float> _Ey_simu_init;
    std::vector<float> _Ez_simu_init;
    std::vector<float> _Hx_simu_init;
    std::vector<float> _Hy_simu_init;
    std::vector<float> _Hz_simu_init;
    std::vector<float> _Ex_simu_sh;
    std::vector<float> _Ey_simu_sh;
    std::vector<float> _Ez_simu_sh;
    std::vector<float> _Hx_simu_sh;
    std::vector<float> _Hy_simu_sh;
    std::vector<float> _Hz_simu_sh;

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

    // FDTD parameters 
    const float um = 1.0f;
    const float nm = um / 1.0e3;
    const float SOURCE_WAVELENGTH = 1 * um;
    const float SOURCE_FREQUENCY = c0 / SOURCE_WAVELENGTH;  // frequency of the source
    const float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;
    const float _dx = SOURCE_WAVELENGTH / 10;
    // const float dt = 0.56f * _dx / c0;  // courant factor: c * dt < dx / sqrt(3)
    const float dt = 0.05f;  // courant factor: c * dt < dx / sqrt(3)
    float J_source_amp = 5e4;
    float M_source_amp = J_source_amp * std::pow(eta0, 3.0);
    float freq_sigma = 0.05 * SOURCE_FREQUENCY;
    float t_sigma = 1 / freq_sigma / (2 * PI); // used to calculate Gaussian pulse width
    float t_peak = 5 * t_sigma;
};  

void gDiamond::update_FDTD_seq_check_result(size_t num_timesteps) { // only use for result checking

  // create temporary E and H for experiments
  std::vector<float> Ex_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ey_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Ez_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hx_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hy_temp(_Nx * _Ny * _Nz, 0);
  std::vector<float> Hz_temp(_Nx * _Ny * _Nz, 0);

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t total_cal = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(size_t k=1; k<_Nz-1; k++) {
      for(size_t j=1; j<_Ny-1; j++) {
        for(size_t i=1; i<_Nx-1; i++) {
          size_t idx = i + j*_Nx + k*(_Nx*_Ny);
          
          total_cal++;

          Ex_temp[idx] = _Cax[idx] * Ex_temp[idx] + _Cbx[idx] *
            ((Hz_temp[idx] - Hz_temp[idx - _Nx]) - (Hy_temp[idx] - Hy_temp[idx - _Nx * _Ny]) - _Jx[idx] * _dx);
          Ey_temp[idx] = _Cay[idx] * Ey_temp[idx] + _Cby[idx] *
            ((Hx_temp[idx] - Hx_temp[idx - _Nx * _Ny]) - (Hz_temp[idx] - Hz_temp[idx - 1]) - _Jy[idx] * _dx);
          Ez_temp[idx] = _Caz[idx] * Ez_temp[idx] + _Cbz[idx] *
            ((Hy_temp[idx] - Hy_temp[idx - 1]) - (Hx_temp[idx] - Hx_temp[idx - _Nx]) - _Jz[idx] * _dx);

        }
      }
    }

    // std::cout << "Ex_temp = \n";
    // for(size_t k=1; k<_Nz-1; k++) {
    //   for(size_t j=1; j<_Ny-1; j++) {
    //     for(size_t i=1; i<_Nx-1; i++) {
    //       size_t idx = i + j*_Nx + k*(_Nx*_Ny);
    //       if(Ex_temp[idx] != 0) { 
    //         std::cout << "t = " << t << ", ";
    //         std::cout << "(x, y, z) = " << i << ", " << j << ", " << k << ", ";
    //         std::cout << "Ex_temp[idx] = " << Ex_temp[idx] << "\n";
    //       }
    //     }
    //   }
    // }

    // update H
    for(size_t k=1; k<_Nz-1; k++) {
      for(size_t j=1; j<_Ny-1; j++) {
        for(size_t i=1; i<_Nx-1; i++) {
          size_t idx = i + j*_Nx + k*(_Nx*_Ny);

          total_cal++;

          Hx_temp[idx] = _Dax[idx] * Hx_temp[idx] + _Dbx[idx] *
            ((Ey_temp[idx + _Nx * _Ny] - Ey_temp[idx]) - (Ez_temp[idx + _Nx] - Ez_temp[idx]) - _Mx[idx] * _dx);
          Hy_temp[idx] = _Day[idx] * Hy_temp[idx] + _Dby[idx] *
            ((Ez_temp[idx + 1] - Ez_temp[idx]) - (Ex_temp[idx + _Nx * _Ny] - Ex_temp[idx]) - _My[idx] * _dx);
          Hz_temp[idx] = _Daz[idx] * Hz_temp[idx] + _Dbz[idx] *
            ((Ex_temp[idx + _Nx] - Ex_temp[idx]) - (Ey_temp[idx + 1] - Ey_temp[idx]) - _Mz[idx] * _dx);

          // if(t == 1 && i == 13 && j == 1 && k == 2) {
          //   std::cout << "here, Hz_temp[idx] = " << Hz_temp[idx]
          //             << ", Ex_temp[idx + _Nx] = " << Ex_temp[idx + _Nx]
          //             << ", Ex_temp[idx] = " << Ex_temp[idx]
          //             << ", Ey_temp[idx + 1] = " << Ey_temp[idx + 1]
          //             << ", Ey_temp[idx] = " << Ey_temp[idx]
          //             << ", _Mz[idx] = " << _Mz[idx] 
          //             << "\n";
          // }

        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "seq runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "seq performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";
  std::cout << "seq total calculations: " << total_cal << "\n";

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_seq[i] = Ex_temp[i];
    _Ey_seq[i] = Ey_temp[i];
    _Ez_seq[i] = Ez_temp[i];
    _Hx_seq[i] = Hx_temp[i];
    _Hy_seq[i] = Hy_temp[i];
    _Hz_seq[i] = Hz_temp[i];
  }
}

void gDiamond::update_FDTD_seq_figures(size_t num_timesteps) {

  if (std::filesystem::create_directory("seq_figures")) {
      std::cerr << "seq_figures created successfully. " << std::endl;
  } else {
      std::cerr << "failed to create seq_figures or it already exists." << std::endl;
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

  std::chrono::duration<double> seq_runtime(0);

  for(size_t t=0; t<num_timesteps; t++) {

    auto start = std::chrono::high_resolution_clock::now();
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    // std::cout << "seq: Mz_value = " << Mz_value << "\n";
    _Mz[_source_idx] = Mz_value;

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

    auto end = std::chrono::high_resolution_clock::now();

    seq_runtime += end-start;

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

      snprintf(field_filename, sizeof(field_filename), "seq_figures/Hz_seq_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }

  }
  std::cout << "seq runtime (excluding figures output): " << seq_runtime.count() << "s\n"; 
  std::cout << "seq performance (excluding figures output): " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / seq_runtime.count() << "Mcells/s\n";

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_seq[i] = Ex_temp[i];
    _Ey_seq[i] = Ey_temp[i];
    _Ez_seq[i] = Ez_temp[i];
    _Hx_seq[i] = Hx_temp[i];
    _Hy_seq[i] = Hy_temp[i];
    _Hz_seq[i] = Hz_temp[i];
  }

}

void gDiamond::update_FDTD_seq(size_t num_timesteps) {

  /*
  if (std::filesystem::create_directory("seq_figures")) {
      std::cerr << "seq_figures created successfully. " << std::endl;
  } else {
      std::cerr << "failed to create seq_figures or it already exists." << std::endl;
      std::exit(EXIT_FAILURE);
  }
  */

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
    // std::cout << "seq: Mz_value = " << Mz_value << "\n";
    _Mz[_source_idx] = Mz_value;

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

    /*
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

      snprintf(field_filename, sizeof(field_filename), "seq_figures/Hz_seq_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }
    */

  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "seq runtime (excluding figures output): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "seq performance (excluding figures output): " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_seq[i] = Ex_temp[i];
    _Ey_seq[i] = Ey_temp[i];
    _Ez_seq[i] = Ez_temp[i];
    _Hx_seq[i] = Hx_temp[i];
    _Hy_seq[i] = Hy_temp[i];
    _Hz_seq[i] = Hz_temp[i];
  }
}

void gDiamond::update_FDTD_seq_test(size_t num_timesteps) {

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
  size_t BLT = 10;
  for(size_t tt=0; tt<num_timesteps/BLT; tt++) {
    for(size_t t=0; t<BLT; t+=1) { 

      float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * (t+tt*BLT) * dt);
      _Mz[_source_idx] = Mz_value;

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
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "seq test runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_seq[i] = Ex_temp[i];
    _Ey_seq[i] = Ey_temp[i];
    _Ez_seq[i] = Ez_temp[i];
    _Hx_seq[i] = Hx_temp[i];
    _Hy_seq[i] = Hy_temp[i];
    _Hz_seq[i] = Hz_temp[i];
  }
}

bool gDiamond::check_correctness_gpu_2D() {
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

bool gDiamond::check_correctness_gpu() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_gpu_bl[i] - _Ex_gpu[i]) >= 1e-8 ||
       fabs(_Ey_gpu_bl[i] - _Ey_gpu[i]) >= 1e-8 ||
       fabs(_Ez_gpu_bl[i] - _Ez_gpu[i]) >= 1e-8 ||
       fabs(_Hx_gpu_bl[i] - _Hx_gpu[i]) >= 1e-8 ||
       fabs(_Hy_gpu_bl[i] - _Hy_gpu[i]) >= 1e-8 ||
       fabs(_Hz_gpu_bl[i] - _Hz_gpu[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
} 

bool gDiamond::check_correctness_gpu_shmem() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_gpu_bl[i] - _Ex_gpu_shEH[i]) >= 1e-8 ||
       fabs(_Ey_gpu_bl[i] - _Ey_gpu_shEH[i]) >= 1e-8 ||
       fabs(_Ez_gpu_bl[i] - _Ez_gpu_shEH[i]) >= 1e-8 ||
       fabs(_Hx_gpu_bl[i] - _Hx_gpu_shEH[i]) >= 1e-8 ||
       fabs(_Hy_gpu_bl[i] - _Hy_gpu_shEH[i]) >= 1e-8 ||
       fabs(_Hz_gpu_bl[i] - _Hz_gpu_shEH[i]) >= 1e-8
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

bool gDiamond::check_correctness_omp_dt() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_seq[i] - _Ex_omp_dt[i]) >= 1e-8 ||
       fabs(_Ey_seq[i] - _Ey_omp_dt[i]) >= 1e-8 ||
       fabs(_Ez_seq[i] - _Ez_omp_dt[i]) >= 1e-8 ||
       fabs(_Hx_seq[i] - _Hx_omp_dt[i]) >= 1e-8 ||
       fabs(_Hy_seq[i] - _Hy_omp_dt[i]) >= 1e-8 ||
       fabs(_Hz_seq[i] - _Hz_omp_dt[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
} 

bool gDiamond::check_correctness_simu() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_seq[i] - _Ex_simu[i]) >= 1e-8 ||
       fabs(_Ey_seq[i] - _Ey_simu[i]) >= 1e-8 ||
       fabs(_Ez_seq[i] - _Ez_simu[i]) >= 1e-8 ||
       fabs(_Hx_seq[i] - _Hx_simu[i]) >= 1e-8 ||
       fabs(_Hy_seq[i] - _Hy_simu[i]) >= 1e-8 ||
       fabs(_Hz_seq[i] - _Hz_simu[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
} 

bool gDiamond::check_correctness_simu_shmem() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_simu_sh[i] - _Ex_simu[i]) >= 1e-8 ||
       fabs(_Ey_simu_sh[i] - _Ey_simu[i]) >= 1e-8 ||
       fabs(_Ez_simu_sh[i] - _Ez_simu[i]) >= 1e-8 ||
       fabs(_Hx_simu_sh[i] - _Hx_simu[i]) >= 1e-8 ||
       fabs(_Hy_simu_sh[i] - _Hy_simu[i]) >= 1e-8 ||
       fabs(_Hz_simu_sh[i] - _Hz_simu[i]) >= 1e-8
    ) {
      std::cout << "_Ex_simu_sh[i] = " << _Ex_simu_sh[i] << ", _Ex_simu[i] = " << _Ex_simu[i] << "\n";
      std::cout << "_Ey_simu_sh[i] = " << _Ey_simu_sh[i] << ", _Ey_simu[i] = " << _Ey_simu[i] << "\n";
      std::cout << "_Ez_simu_sh[i] = " << _Ez_simu_sh[i] << ", _Ez_simu[i] = " << _Ez_simu[i] << "\n";
      std::cout << "_Hx_simu_sh[i] = " << _Hx_simu_sh[i] << ", _Hx_simu[i] = " << _Hx_simu[i] << "\n";
      std::cout << "_Hy_simu_sh[i] = " << _Hy_simu_sh[i] << ", _Hy_simu[i] = " << _Hy_simu[i] << "\n";
      std::cout << "_Hz_simu_sh[i] = " << _Hz_simu_sh[i] << ", _Hz_simu[i] = " << _Hz_simu[i] << "\n";
      correct = false;
      break;
    }
  }

  return correct;
} 

bool gDiamond::check_correctness_rep() {
  bool correct = true;

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    if(fabs(_Ex_gpu[i] - _Ex_simu[i]) >= 1e-8 ||
       fabs(_Ey_gpu[i] - _Ey_simu[i]) >= 1e-8 ||
       fabs(_Ez_gpu[i] - _Ez_simu[i]) >= 1e-8 ||
       fabs(_Hx_gpu[i] - _Hx_simu[i]) >= 1e-8 ||
       fabs(_Hy_gpu[i] - _Hy_simu[i]) >= 1e-8 ||
       fabs(_Hz_gpu[i] - _Hz_simu[i]) >= 1e-8
    ) {
      correct = false;
      break;
    }
  }

  return correct;
}

void gDiamond::_get_indices_and_ranges(size_t BLX, size_t BLT, size_t Nx,
                                       std::vector<std::vector<size_t>>& mountain_indices,
                                       std::vector<std::vector<size_t>>& mountain_ranges,
                                       std::vector<std::vector<size_t>>& valley_indices,
                                       std::vector<std::vector<size_t>>& valley_ranges
                                      ) {

  // here we always set a mountain at the beginning of space
  // base on that 1st row is E 

  size_t mountain_bottom = BLX;
  size_t valley_top = BLX - 2*(BLT - 1) - 1;
  size_t two_tiles = mountain_bottom + valley_top;
  // fill up mountain_indices
  mountain_indices.resize(BLT*2);
  size_t row = 0;
  for(size_t t=0; t<BLT; t++) {
    for(size_t index=t; index<Nx; index+=two_tiles) {
      mountain_indices[row].push_back(index);
      mountain_indices[row+1].push_back(index);
    }
    row += 2;
  }
  // fill up mountain_ranges 
  mountain_ranges.resize(BLT*2);
  size_t range = BLX;
  for(size_t row=0; row<BLT*2; row++) {
    for(auto index : mountain_indices[row]) {
      if(index + range < Nx) {
        mountain_ranges[row].push_back(range);
      } 
      else {
        mountain_ranges[row].push_back(Nx - index);
      }
    }
    --range;
  }

  // fill up valley_indices and valley_ranges
  valley_indices.resize(BLT*2);
  valley_ranges.resize(BLT*2);
  for(size_t row=0; row<BLT*2; row++) {
    size_t current = 0; // Pointer for AB_mix
    for (size_t i = 0; i < mountain_indices[row].size(); ++i) {
      size_t mountain_start = mountain_indices[row][i];
      size_t mountain_length = mountain_ranges[row][i];

      if (current < mountain_start) {
        valley_indices[row].push_back(current);
        valley_ranges[row].push_back(mountain_start - current);
      }

      current = mountain_start + mountain_length;
    }

    if (current < Nx) {
      valley_indices[row].push_back(current);
      valley_ranges[row].push_back(Nx - current);
    }
  }

}

std::vector<size_t> gDiamond::_set_ranges_X(size_t row, size_t xx, size_t phase) {

  // results = {x_head, x_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 2 || phase == 3 || phase == 6) {
    results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(1));
    results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-2);
  }
  else if(phase == 1 || phase == 4 || phase == 5 || phase == 7) {
    results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(1));
    results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-2);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}

std::vector<size_t> gDiamond::_set_ranges_Y(size_t row, size_t yy, size_t phase) {

  // results = {y_head, y_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 1 || phase == 3 || phase == 5) {
    results[0] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(1));
    results[1] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-2);
  }
  else if(phase == 2 || phase == 4 || phase == 6 || phase == 7) {
    results[0] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(1));
    results[1] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-2);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}

std::vector<size_t> gDiamond::_set_ranges_Z(size_t row, size_t zz, size_t phase) {

  // results = {z_head, z_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 1 || phase == 2 || phase == 4) {
    results[0] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(1));
    results[1] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-2);
  }
  else if(phase == 3 || phase == 5 || phase == 6 || phase == 7) {
    results[0] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(1));
    results[1] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-2);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}

std::vector<size_t> gDiamond::_set_ranges_X_gpu(size_t row, size_t xx, size_t phase) {

  // results = {x_head, x_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 2 || phase == 3 || phase == 6) {
    results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(0));
    results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-1);
  }
  else if(phase == 1 || phase == 4 || phase == 5 || phase == 7) {
    results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(0));
    results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-1);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}

std::vector<size_t> gDiamond::_set_ranges_Y_gpu(size_t row, size_t yy, size_t phase) {

  // results = {y_head, y_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 1 || phase == 3 || phase == 5) {
    results[0] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(0));
    results[1] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-1);
  }
  else if(phase == 2 || phase == 4 || phase == 6 || phase == 7) {
    results[0] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(0));
    results[1] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-1);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}

std::vector<size_t> gDiamond::_set_ranges_Z_gpu(size_t row, size_t zz, size_t phase) {

  // results = {z_head, z_tail}
  std::vector<size_t> results(2); 
  if(phase == 0 || phase == 1 || phase == 2 || phase == 4) {
    results[0] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(0));
    results[1] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-1);
  }
  else if(phase == 3 || phase == 5 || phase == 6 || phase == 7) {
    results[0] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(0));
    results[1] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-1);
  }
  else {
    std::cerr << "error: phase wrong\n";
    std::exit(EXIT_FAILURE);
  }

  return results;
}



std::vector<int> gDiamond::_set_ranges(size_t row, size_t xx, size_t yy, size_t zz, size_t phase) {
 
  // results = {x_head, x_tail, y_head, y_tail, z_head, z_tail}
  std::vector<int> results(6, -1); 
  switch(phase) {
    case 0: // phase 1: mountains on X, mountains on Y, mountains on Z
      results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 1: // phase 2: valleys on X, mountains on Y, mountains on Z
      results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 2: // phase 3: mountains on X, valleys on Y, mountains on Z
      results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 3: // phase 4: mountains on X, mountains on Y, valleys on Z
      results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 4: // phase 5: valleys on X, valleys on Y, mountains on Z
      results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_mountain_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_mountain_indices_Z[row][zz] + _mountain_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 5: // phase 6: valleys on X, mountains on Y, valleys on Z
      results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_mountain_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_mountain_indices_Y[row][yy] + _mountain_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 6: // phase 7: mountains on X, valleys on Y, valleys on Z
      results[0] = std::max(_mountain_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_mountain_indices_X[row][xx] + _mountain_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-2);
      break;
    case 7: // phase 8: valleys on X, valleys on Y, valleys on Z
      results[0] = std::max(_valley_indices_X[row][xx], static_cast<size_t>(1));
      results[1] = std::min(_valley_indices_X[row][xx] + _valley_ranges_X[row][xx] - 1, _Nx-2);
      results[2] = std::max(_valley_indices_Y[row][yy], static_cast<size_t>(1));
      results[3] = std::min(_valley_indices_Y[row][yy] + _valley_ranges_Y[row][yy] - 1, _Ny-2);
      results[4] = std::max(_valley_indices_Z[row][zz], static_cast<size_t>(1));
      results[5] = std::min(_valley_indices_Z[row][zz] + _valley_ranges_Z[row][zz] - 1, _Nz-2);
      break;
  }

  for(size_t i=0; i<6; i++) {
    if(results[i] < 0) {
      std::cerr << "error: x_head/x_tail wrong\n";
      std::exit(EXIT_FAILURE);
    }
  }

  return results;
}

std::vector<int> gDiamond::_set_num_tiles(size_t row, size_t phase) {

  // results = {x_ntiles, y_ntiles, z_tiles}
  std::vector<int> results(3, -1); 

  switch(phase) {
    case 0: // phase 1: mountains on X, mountains on Y, mountains on Z
      results[0] = _mountain_indices_X[row].size(); 
      results[1] = _mountain_indices_Y[row].size(); 
      results[2] = _mountain_indices_Z[row].size(); 
      break;
    case 1: // phase 2: valleys on X, mountains on Y, mountains on Z
      results[0] = _valley_indices_X[row].size(); 
      results[1] = _mountain_indices_Y[row].size(); 
      results[2] = _mountain_indices_Z[row].size(); 
      break;
    case 2: // phase 3: mountains on X, valleys on Y, mountains on Z
      results[0] = _mountain_indices_X[row].size(); 
      results[1] = _valley_indices_Y[row].size(); 
      results[2] = _mountain_indices_Z[row].size(); 
      break;
    case 3: // phase 4: mountains on X, mountains on Y, valleys on Z
      results[0] = _mountain_indices_X[row].size(); 
      results[1] = _mountain_indices_Y[row].size(); 
      results[2] = _valley_indices_Z[row].size(); 
      break;
    case 4: // phase 5: valleys on X, valleys on Y, mountains on Z
      results[0] = _valley_indices_X[row].size(); 
      results[1] = _valley_indices_Y[row].size(); 
      results[2] = _mountain_indices_Z[row].size(); 
      break;
    case 5: // phase 6: valleys on X, mountains on Y, valleys on Z
      results[0] = _valley_indices_X[row].size(); 
      results[1] = _mountain_indices_Y[row].size(); 
      results[2] = _valley_indices_Z[row].size(); 
      break;
    case 6: // phase 7: mountains on X, valleys on Y, valleys on Z
      results[0] = _mountain_indices_X[row].size(); 
      results[1] = _valley_indices_Y[row].size(); 
      results[2] = _valley_indices_Z[row].size(); 
      break;
    case 7: // phase 8: valleys on X, valleys on Y, valleys on Z
      results[0] = _valley_indices_X[row].size(); 
      results[1] = _valley_indices_Y[row].size(); 
      results[2] = _valley_indices_Z[row].size(); 
      break;
  }

  for(size_t i=0; i<3; i++) {
    if(results[i] < 0) {
      std::cerr << "error: number of tiles wrong\n";
      std::exit(EXIT_FAILURE);
    }
  }

  return results;
}

void gDiamond::_setup_diamond_tiling(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t max_phases) {

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

  /*
  std::cout << "_Entiles_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto n : _Entiles_phases_X[0]) {
    std::cout << n << " ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto n : _Entiles_phases_X[1]) {
    std::cout << n << " ";
  }
  std::cout << "]\n";

  std::cout << "_Eranges_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto range : _Eranges_phases_X[0][0]) { // that is for mountain heads and tails
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto range : _Eranges_phases_X[1][0]) {
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";

  std::cout << "_Hranges_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto range : _Hranges_phases_X[0][BLT-1]) {
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto range : _Hranges_phases_X[1][BLT-1]) { // that is for valley heads and tails
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  */
}

void gDiamond::_setup_diamond_tiling_gpu(size_t BLX, size_t BLY, size_t BLZ, size_t BLT, size_t max_phases) {

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
        std::vector<size_t> head_tail = _set_ranges_X_gpu(E_row, xx, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_X_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_X_size_adjust;
        }
      }
      _Entiles_phases_X[phase][t] -= Entiles_onephase_X_size_adjust;
      for(size_t yy=0; yy<_Entiles_phases_Y[phase][t]; yy++) {
        std::vector<size_t> head_tail = _set_ranges_Y_gpu(E_row, yy, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_Y_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_Y_size_adjust;
        }
      }
      _Entiles_phases_Y[phase][t] -= Entiles_onephase_Y_size_adjust;
      for(size_t zz=0; zz<_Entiles_phases_Z[phase][t]; zz++) {
        std::vector<size_t> head_tail = _set_ranges_Z_gpu(E_row, zz, phase);
        if(head_tail[0] <= head_tail[1]) {
          Eranges_onephase_Z_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Entiles_onephase_Z_size_adjust;
        }
      }
      _Entiles_phases_Z[phase][t] -= Entiles_onephase_Z_size_adjust;
      for(size_t xx=0; xx<_Hntiles_phases_X[phase][t]; xx++) {
        std::vector<size_t> head_tail = _set_ranges_X_gpu(H_row, xx, phase);
        if(head_tail[0] <= head_tail[1]) {
          Hranges_onephase_X_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Hntiles_onephase_X_size_adjust;
        }
      }
      _Hntiles_phases_X[phase][t] -= Hntiles_onephase_X_size_adjust;
      for(size_t yy=0; yy<_Hntiles_phases_Y[phase][t]; yy++) {
        std::vector<size_t> head_tail = _set_ranges_Y_gpu(H_row, yy, phase);
        if(head_tail[0] <= head_tail[1]) {
          Hranges_onephase_Y_onet.push_back(std::make_pair(head_tail[0], head_tail[1]));
        }
        else{
          ++Hntiles_onephase_Y_size_adjust;
        }
      }
      _Hntiles_phases_Y[phase][t] -= Hntiles_onephase_Y_size_adjust;
      for(size_t zz=0; zz<_Hntiles_phases_Z[phase][t]; zz++) {
        std::vector<size_t> head_tail = _set_ranges_Z_gpu(H_row, zz, phase);
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

  /*
  std::cout << "_Entiles_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto n : _Entiles_phases_X[0]) {
    std::cout << n << " ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto n : _Entiles_phases_X[1]) {
    std::cout << n << " ";
  }
  std::cout << "]\n";

  std::cout << "_Eranges_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto range : _Eranges_phases_X[0][0]) { // that is for mountain heads and tails
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto range : _Eranges_phases_X[1][0]) {
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";

  std::cout << "_Hranges_phases_X = \n";
  std::cout << "phase 1 = [";
  for(auto range : _Hranges_phases_X[0][BLT-1]) {
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  std::cout << "phase 2 = [";
  for(auto range : _Hranges_phases_X[1][BLT-1]) { // that is for valley heads and tails
    std::cout << "(" << range.first << ", " << range.second << ") ";
  }
  std::cout << "]\n";
  */
}

void gDiamond::print_results() {

  std::cout << "Ex_seq = \n";
  for(size_t k=1; k<_Nz-1; k++) {
    for(size_t j=1; j<_Ny-1; j++) {
      for(size_t i=1; i<_Nx-1; i++) {
        size_t idx = i + j*_Nx + k*(_Nx*_Ny);
        if(_Ex_seq[idx] != 0) { 
          std::cout << "(x, y, z) = " << i << ", " << j << ", " << k << ", ";
          std::cout << "Ex_seq[idx] = " << _Ex_seq[idx] << "\n";
        }
      }
    }
  }

  std::cout << "Ex_simu = \n";
  for(size_t k=1; k<_Nz-1; k++) {
    for(size_t j=1; j<_Ny-1; j++) {
      for(size_t i=1; i<_Nx-1; i++) {
        size_t idx = i + j*_Nx + k*(_Nx*_Ny);
        if(_Ex_simu[idx] != 0) { 
          std::cout << "(x, y, z) = " << i << ", " << j << ", " << k << ", ";
          std::cout << "Ex_simu[idx] = " << _Ex_simu[idx] << "\n";
        }
      }
    }
  }

  /*
  std::cout << "Ey_seq = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Ey_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Ey_gpu = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Ey_gpu[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Ez_seq = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Ez_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Ez_gpu = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Ez_gpu[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hx_seq = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hx_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hx_gpu = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hx_gpu[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hy_seq = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hy_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hy_gpu = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hy_gpu[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hz_seq = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hz_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Hz_gpu = ";
  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    std::cout << _Hz_gpu[i] << " ";
  }
  std::cout << "\n";
  */
}

} // end of namespace gdiamond

#endif




























