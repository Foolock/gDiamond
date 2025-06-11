#ifndef GDIAMOND_GPU_MM_VER5_CUH
#define GDIAMOND_GPU_MM_VER5_CUH

#include "gdiamond.hpp"
#include "kernels_mm_ver5.cuh"
#include <cuda_runtime.h>

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

#define SWAP_PTR(a, b) do { auto _tmp = (a); (a) = (b); (b) = _tmp; } while (0)

namespace gdiamond {

void gDiamond::_updateEH_mix_mapping_ver5(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
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
                                          size_t grid_size) {

  for(size_t block_id = 0; block_id < grid_size; block_id++) {

    // indices to use
    int local_x, local_y, local_z;
    int global_x, global_y, global_z;
    int H_shared_x, H_shared_y, H_shared_z;
    int E_shared_x, E_shared_y, E_shared_z;
    int global_idx;
    int H_shared_idx;
    int E_shared_idx;
    
    const int supertile_id = block_id / num_subtiles;
    const int subtile_id = block_id % num_subtiles;

    // first find the supertile this block belongs to
    const int super_xx = supertile_id % xx_num;
    const int super_yy = (supertile_id / xx_num) % yy_num;
    const int super_zz = supertile_id / (xx_num * yy_num);

    // second find the subtile this block belongs to
    // sub_xx == 0 means it is replication, otherwise it is parallelogram part
    // same for Y, Z
    Pt_idx p = hyperplanes[hyperplane_head + subtile_id];
    const int sub_xx = p.x;
    const int sub_yy = p.y;
    const int sub_zz = p.z;

    // decide where to load and store 
    // Golden rule:
    // When loading, in 3D space, as long as there is one dimension that contains a parallelogram tile, then this Halo load needs to load from temp.
    // When storing, in 3D space, as long as there is one dimension that contains a replication tile, then this data is stored to temp.
    const bool load_from_temp = (sub_xx > 0 || sub_yy > 0 || sub_zz > 0)? true : false;
    const bool store_to_temp = (sub_xx == 0 || sub_yy == 0 || sub_zz == 0)? true : false;

    // decide heads according to supertile and subtile index
    const int xx_head = xx_heads[super_xx] + (sub_xx > 0) * (BLX_R + (sub_xx - 1) * BLX_P);
    const int yy_head = yy_heads[super_yy] + sub_yy * BLY_R; // assume BLY_R = BLY_P
    const int zz_head = zz_heads[super_zz] + sub_zz * BLZ_R; // assume BLZ_R = BLZ_P

    if(super_xx == 0 && super_yy == 0 && super_zz == 0) {
      std::cout << "sub_xx = " << sub_xx 
                << ", sub_yy = " << sub_yy 
                << ", sub_zz = " << sub_zz
                << ", xx_head = " << xx_head
                << ", yy_head = " << yy_head
                << ", zz_head = " << zz_head
                << "\n";
    }

    // declare shared memory
    // parallelogram calculation used more shared memory than replication calculation
    float Hx_shmem[H_SHX_V5 * H_SHY_V5 * H_SHZ_V5];
    float Hy_shmem[H_SHX_V5 * H_SHY_V5 * H_SHZ_V5];
    float Hz_shmem[H_SHX_V5 * H_SHY_V5 * H_SHZ_V5];
    float Ex_shmem[E_SHX_V5 * E_SHY_V5 * E_SHZ_V5];
    float Ey_shmem[E_SHX_V5 * E_SHY_V5 * E_SHZ_V5];
    float Ez_shmem[E_SHX_V5 * E_SHY_V5 * E_SHZ_V5];

    // load shared memory
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V5;
      local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
      local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

      // H_shared and E_shared index are set differently when load replication / parallelogram
      H_shared_x = (sub_xx == 0)? local_x + 1 : local_x + 2;
      H_shared_y = (sub_yy == 0)? local_y + 1 : local_y + 2;
      H_shared_z = (sub_zz == 0)? local_z + 1 : local_z + 2;
      E_shared_x = (sub_xx == 0)? local_x : local_x + 2;
      E_shared_y = (sub_yy == 0)? local_y : local_y + 2;
      E_shared_z = (sub_zz == 0)? local_z : local_z + 2;

      // global index is the same with head index as base 
      global_x = xx_head + local_x;
      global_y = yy_head + local_y;
      global_z = zz_head + local_z;

      // load core ---------------------------------------------
      // Difference: if load replication (sub == 0), ONLY in X dimension, loadcore_tail_X need to be xx_head + BLX_R - 1 
      //             loadcore_head_X is the same for replication / parallelogram, no need to specify
      //             For Y, Z dimension, loadcore_head and loadcore_tail is always the same, no need to specify
      int loadcore_tail_X = (sub_xx == 0)? xx_head + BLX_R - 1 : xx_head + BLX_P - 1; 
      H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;
      E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;
      global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      if(global_x <= loadcore_tail_X) {

        // if(local_y == 0 && local_z == 0) {
        //   std::cout << "loading global_x = " << global_x << "\n";
        // }

        Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
        Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
      }

      // load H Halo ---------------------------------------------
      // Difference: if load replication (sub == 0), only one extra H Halo needs to be loaded 
      //             if load parallelogram (sub > 0), two extra H Halo need to be loaded
      if ((sub_xx == 0 && local_x == 0) || (sub_xx > 0 && local_x < 2)) {
        int halo_x = (sub_xx == 0) ? 0 : local_x;
        int global_x_halo = (sub_xx == 0)
                            ? xx_head + halo_x - 1
                            : xx_head + halo_x - 2;

        // if(local_y == 0 && local_z == 0) {
        //   std::cout << "local_x = " << local_x << ", halo_x = " << halo_x << ", global_x_halo = " << global_x_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = halo_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        Hx_shmem[H_shared_idx] = (load_from_temp)? Hx_pad_temp[global_idx] : Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = (load_from_temp)? Hy_pad_temp[global_idx] : Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = (load_from_temp)? Hz_pad_temp[global_idx] : Hz_pad_src[global_idx];
      }
      if((sub_yy == 0 && local_y == 0) || (sub_yy > 0 && local_y < 2)) {
        int halo_y = (sub_yy == 0) ? 0 : local_y;
        int global_y_halo = (sub_yy == 0)
                            ? yy_head + halo_y - 1
                            : yy_head + halo_y - 2;

        // if(local_x == 0 && local_z == 0) {
        //   std::cout << "local_y = " << local_y << ", halo_y = " << halo_y << ", global_y_halo = " << global_y_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + halo_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        Hx_shmem[H_shared_idx] = (load_from_temp)? Hx_pad_temp[global_idx] : Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = (load_from_temp)? Hy_pad_temp[global_idx] : Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = (load_from_temp)? Hz_pad_temp[global_idx] : Hz_pad_src[global_idx];
      }
      if((sub_zz == 0 && local_z == 0) || (sub_zz > 0 && local_z < 2)) {
        int halo_z = (sub_zz == 0) ? 0 : local_z;
        int global_z_halo = (sub_zz == 0)
                            ? zz_head + halo_z - 1
                            : zz_head + halo_z - 2;

        // if(local_x == 0 && local_y == 0) {
        //   std::cout << "local_z = " << local_z << ", halo_z = " << halo_z << ", global_z_halo = " << global_z_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + halo_z * H_SHX_V5 * H_SHY_V5;

        Hx_shmem[H_shared_idx] = (load_from_temp)? Hx_pad_temp[global_idx] : Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = (load_from_temp)? Hy_pad_temp[global_idx] : Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = (load_from_temp)? Hz_pad_temp[global_idx] : Hz_pad_src[global_idx];
      }

      // load E Halo ---------------------------------------------
      // Difference: if load replication (sub == 0), no E Halo need to be loaded
      //             if load parallelogram (sub > 0), two extra E Halo need to be loaded 
      //             Since E Halo won't happen when loading replication, each load must come from temp.
      if(sub_xx > 0 && local_x >= NTX_MM_V5 - 2) {
        int halo_x = local_x - NTX_MM_V5 + 2;
        int global_x_halo = xx_head + halo_x - 2;

        // if(local_y == 0 && local_z == 0) {
        //   std::cout << "local_x = " << local_x << ", halo_x = " << halo_x << ", global_x_halo = " << global_x_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        E_shared_idx = halo_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;

        Ex_shmem[E_shared_idx] = Ex_pad_temp[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_temp[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_temp[global_idx];
      }
      if(sub_yy > 0 && local_y >= NTY_MM_V5 - 2) {
        int halo_y = local_y - NTY_MM_V5 + 2;
        int global_y_halo = yy_head + halo_y - 2;

        // if(local_x == 0 && local_z == 0) {
        //   std::cout << "local_y = " << local_y << ", halo_y = " << halo_y << ", global_y_halo = " << global_y_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
        E_shared_idx = E_shared_x + halo_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;

        Ex_shmem[E_shared_idx] = Ex_pad_temp[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_temp[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_temp[global_idx];
      }
      if(sub_zz > 0 && local_z >= NTZ_MM_V5 - 2) {
        int halo_z = local_z - NTZ_MM_V5 + 2;
        int global_z_halo = zz_head + halo_z - 2;

        // if(local_x == 0 && local_y == 0) {
        //   std::cout << "local_z = " << local_z << ", halo_z = " << halo_z << ", global_z_halo = " << global_z_halo << ", ";
        //   if(load_from_temp) std::cout << "load from temp.\n";
        //   else std::cout << "load from src.\n";
        // }

        global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + halo_z * E_SHX_V5 * E_SHY_V5;

        Ex_shmem[E_shared_idx] = Ex_pad_temp[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_temp[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_temp[global_idx];
      }
    }

    int calE_head_X;
    int calE_tail_X;
    int calH_head_X;
    int calH_tail_X;
    int calE_head_Y;
    int calE_tail_Y;
    int calH_head_Y;
    int calH_tail_Y;
    int calE_head_Z;
    int calE_tail_Z;
    int calH_head_Z;
    int calH_tail_Z;
    // Convert sub_xx == 0 into a binary flag
    int is_rep_x = (sub_xx == 0); // 1 if replication, 0 otherwise
    int is_rep_y = (sub_yy == 0);
    int is_rep_z = (sub_zz == 0);
    // Signed direction: +1 if replication, -1 if parallelogram
    int sgn_x = 2 * is_rep_x - 1;  // +1 or -1
    int sgn_y = 2 * is_rep_y - 1;
    int sgn_z = 2 * is_rep_z - 1;

    // calculation for the first 2 timesteps
    // Difference: if replication (sub == 0), calculation bounds follow mountain tiling
    //             if parallelogram (sub > 0), calculation bounds follow parallelogram tiling
    //             There is no difference when actually calculating
    for(int t = 0; t < BLT_MM_V5 / 2; t++) {

      calE_head_X = xx_head + sgn_x * t;
      calE_head_Y = yy_head + sgn_y * t;
      calE_head_Z = zz_head + sgn_z * t;
      calE_tail_X = is_rep_x * (xx_head + BLX_R - 1 - t) +
                   (1 - is_rep_x) * (calE_head_X + BLX_P - 1);
      calE_tail_Y = is_rep_y * (yy_head + BLY_R - 1 - t) +
                   (1 - is_rep_y) * (calE_head_Y + BLY_P - 1);
      calE_tail_Z = is_rep_z * (zz_head + BLZ_R - 1 - t) +
                   (1 - is_rep_z) * (calE_head_Z + BLZ_P - 1);
      calH_head_X = calE_head_X - (1 - is_rep_x);
      calH_tail_X = calE_tail_X - 1;
      calH_head_Y = calE_head_Y - (1 - is_rep_y);
      calH_tail_Y = calE_tail_Y - 1;
      calH_head_Z = calE_head_Z - (1 - is_rep_z);
      calH_tail_Z = calE_tail_Z - 1;

      // if(super_xx == 0 && super_yy == 0 && super_zz == 0) {
      //   std::cout << "t = " << t << "\n";
      //   std::cout << "calE_head_X = " << calE_head_X << ", calE_tail_X = " << calE_tail_X
      //             << ", calH_head_X = " << calH_head_X << ", calH_tail_X = " << calH_tail_X << "\n";
      //   std::cout << "calE_head_Y = " << calE_head_Y << ", calE_tail_Y = " << calE_tail_Y
      //             << ", calH_head_Y = " << calH_head_Y << ", calH_tail_Y = " << calH_tail_Y << "\n";
      //   std::cout << "calE_head_Z = " << calE_head_Z << ", calE_tail_Z = " << calE_tail_Z
      //             << ", calH_head_Z = " << calH_head_Z << ", calH_tail_Z = " << calH_tail_Z << "\n";
      // }

      // update E
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V5;
        local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
        local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

        // H_shared and E_shared index are set differently when load replication / parallelogram
        // when calculate **E** in parallelogram, offset between local and H/E_shared is 2 - t
        H_shared_x = (sub_xx == 0)? local_x + 1 : local_x + 2 - t;
        H_shared_y = (sub_yy == 0)? local_y + 1 : local_y + 2 - t;
        H_shared_z = (sub_zz == 0)? local_z + 1 : local_z + 2 - t;
        E_shared_x = (sub_xx == 0)? local_x : local_x + 2 - t;
        E_shared_y = (sub_yy == 0)? local_y : local_y + 2 - t;
        E_shared_z = (sub_zz == 0)? local_z : local_z + 2 - t;

        // global index in replication / parallelogram is set differently when calculate **E** replication / parallelogram
        // Difference: if calculate replication (sub == 0), global_x = xx_head + local_x, same for Y, Z 
        //             if calculate parallelogram (sub > 0), global_x = xx_head - t + local_x, same for Y, Z 
        global_x = (sub_xx == 0)? xx_head + local_x : xx_head - t + local_x;
        global_y = (sub_yy == 0)? yy_head + local_y : yy_head - t + local_y;
        global_z = (sub_zz == 0)? zz_head + local_z : zz_head - t + local_z;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        global_idx = (global_x - LEFT_PAD_MM_V5) + (global_y - LEFT_PAD_MM_V5) * Nx + (global_z - LEFT_PAD_MM_V5) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        if(global_x >= 1 + LEFT_PAD_MM_V5 && global_x <= Nx - 2 + LEFT_PAD_MM_V5 &&
           global_y >= 1 + LEFT_PAD_MM_V5 && global_y <= Ny - 2 + LEFT_PAD_MM_V5 &&
           global_z >= 1 + LEFT_PAD_MM_V5 && global_z <= Nz - 2 + LEFT_PAD_MM_V5 &&
           global_x >= calE_head_X && global_x <= calE_tail_X &&
           global_y >= calE_head_Y && global_y <= calE_tail_Y &&
           global_z >= calE_head_Z && global_z <= calE_tail_Z) {

          Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                    ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V5]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5]) - Jx[global_idx] * dx);

          Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                    ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

          Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                    ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V5]) - Jz[global_idx] * dx);

          // if(super_xx == 0 && super_yy == 0 && super_zz == 0 && sub_xx == 1 && sub_yy == 1 && sub_zz == 0) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }

        }
      }

      // update H
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V5;
        local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
        local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

        // H_shared and E_shared index are set differently when load replication / parallelogram
        // when calculate **H** in parallelogram, offset between local and H/E_shared is 1 - t
        H_shared_x = (sub_xx == 0)? local_x + 1 : local_x + 1 - t;
        H_shared_y = (sub_yy == 0)? local_y + 1 : local_y + 1 - t;
        H_shared_z = (sub_zz == 0)? local_z + 1 : local_z + 1 - t;
        E_shared_x = (sub_xx == 0)? local_x : local_x + 1 - t;
        E_shared_y = (sub_yy == 0)? local_y : local_y + 1 - t;
        E_shared_z = (sub_zz == 0)? local_z : local_z + 1 - t;

        // global index in replication / parallelogram is set differently when calculate **H** replication / parallelogram
        // Difference: if calculate replication (sub == 0), global_x = xx_head + local_x, same for Y, Z 
        //             if calculate parallelogram (sub > 0), global_x = xx_head - t + local_x - 1, same for Y, Z 
        global_x = (sub_xx == 0)? xx_head + local_x : xx_head - t + local_x - 1;
        global_y = (sub_yy == 0)? yy_head + local_y : yy_head - t + local_y - 1;
        global_z = (sub_zz == 0)? zz_head + local_z : zz_head - t + local_z - 1;

        global_idx = (global_x - LEFT_PAD_MM_V5) + (global_y - LEFT_PAD_MM_V5) * Nx + (global_z - LEFT_PAD_MM_V5) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        if(global_x >= 1 + LEFT_PAD_MM_V5 && global_x <= Nx - 2 + LEFT_PAD_MM_V5 &&
           global_y >= 1 + LEFT_PAD_MM_V5 && global_y <= Ny - 2 + LEFT_PAD_MM_V5 &&
           global_z >= 1 + LEFT_PAD_MM_V5 && global_z <= Nz - 2 + LEFT_PAD_MM_V5 &&
           global_x >= calH_head_X && global_x <= calH_tail_X &&
           global_y >= calH_head_Y && global_y <= calH_tail_Y &&
           global_z >= calH_head_Z && global_z <= calH_tail_Z) {

          Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                    ((Ey_shmem[E_shared_idx + E_SHX_V5 * E_SHY_V5] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V5] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

          Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                    ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V5 * E_SHY_V5] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

          Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                    ((Ex_shmem[E_shared_idx + E_SHX_V5] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
        
          // if(super_xx == 0 && super_yy == 0 && super_zz == 0 && sub_xx == 1 && sub_yy == 1 && sub_zz == 0) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }

        }
      }
    }

    // load new HALO, evict old data to global memory
    // Since this eviction only happens on the side of the parallelogram,
    // the evict data will all be stored into temp copy 

    // evict old data
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V5;
      local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
      local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

      // H_shared and E_shared index are set differently in replication / parallelogram
      // In eviction, follow the same pattern in shared memory load 
      H_shared_x = (sub_xx == 0)? local_x + 1 : local_x + 2;
      H_shared_y = (sub_yy == 0)? local_y + 1 : local_y + 2;
      H_shared_z = (sub_zz == 0)? local_z + 1 : local_z + 2;
      E_shared_x = (sub_xx == 0)? local_x : local_x + 2;
      E_shared_y = (sub_yy == 0)? local_y : local_y + 2;
      E_shared_z = (sub_zz == 0)? local_z : local_z + 2;

      // global index is the same with head index as base 
      global_x = xx_head + local_x;
      global_y = yy_head + local_y;
      global_z = zz_head + local_z;

      // evict old H ---------------------------------------------
      // For one dimension, it only happens if this dimension is a parallelogram
      if (sub_xx > 0 && local_x < 2) {

        int halo_x = local_x + NTX_MM_V5;
        int global_x_halo = xx_head + halo_x - 2;

        if(local_y == 0 && local_z == 0) {
          std::cout << "evict H: local_x = " << local_x << ", halo_x = " << halo_x << ", global_x_halo = " << global_x_halo << "\n";
        }

        global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = halo_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        Hx_pad_temp[global_idx] = Hx_shmem[H_shared_idx];
        Hy_pad_temp[global_idx] = Hy_shmem[H_shared_idx];
        Hz_pad_temp[global_idx] = Hz_shmem[H_shared_idx];
      }

    }

  }

}



void gDiamond::update_FDTD_mix_mapping_sequential_ver5(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  std::cout << "running update_FDTD_mix_mapping_sequential_ver5...\n";
  std::cout << "supertiles are the larger overlapped mountain tiles. " 
               "Within one supertile there are a bunch of subtiles, "
               "containing one small mountain and a series of parallelograms.\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V5 + RIGHT_PAD_MM_V5;
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V5 + RIGHT_PAD_MM_V5;
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V5 + RIGHT_PAD_MM_V5;
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  /*
   * Though we run each hyperplane with one kernel, there are
   * still conflict. The conflict happens in the storing of 
   * replication subtile and the reading of the last parallelogram subtile
   * from the left large mountain supertile.
   * Consider one dimension, if we store the replication part in-place,
   * when calculating this last parallelogram subtile from the left supertile
   * we will give extra calculation to a data.
   */
  std::vector<float> Ex_pad_src(padded_length, 0);
  std::vector<float> Ey_pad_src(padded_length, 0);
  std::vector<float> Ez_pad_src(padded_length, 0);
  std::vector<float> Hx_pad_src(padded_length, 0);
  std::vector<float> Hy_pad_src(padded_length, 0);
  std::vector<float> Hz_pad_src(padded_length, 0);
  std::vector<float> Ex_pad_temp(padded_length, 0);
  std::vector<float> Ey_pad_temp(padded_length, 0);
  std::vector<float> Ez_pad_temp(padded_length, 0);
  std::vector<float> Hx_pad_temp(padded_length, 0);
  std::vector<float> Hy_pad_temp(padded_length, 0);
  std::vector<float> Hz_pad_temp(padded_length, 0);

  // find hyperplanes of tiles given Nx, Ny, Nz
  std::vector<Pt_idx> hyperplanes;
  std::vector<int> hyperplane_heads;
  std::vector<int> hyperplane_sizes;
  _find_diagonal_hyperplanes(NUM_P_X + 1, NUM_P_Y + 1, NUM_P_Z + 1, 
                             hyperplanes, 
                             hyperplane_heads, 
                             hyperplane_sizes);

  // decide tiling parameters (for supertiles)
  size_t xx_num = Tx;
  size_t yy_num = Ty;
  size_t zz_num = Tz;
  std::vector<int> xx_heads(xx_num, 0); // head indices of big mountains
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);
  
  for(size_t index=0; index<xx_num; index++) {
    xx_heads[index] = (index == 0)? 1 :
                             xx_heads[index-1] + NUM_P_X * BLX_P;
  }

  for(size_t index=0; index<yy_num; index++) {
    yy_heads[index] = (index == 0)? 1 :
                             yy_heads[index-1] + NUM_P_Y * BLY_P;
  }

  for(size_t index=0; index<zz_num; index++) {
    zz_heads[index] = (index == 0)? 1 :
                             zz_heads[index-1] + NUM_P_Z * BLZ_P;
  }

  std::cout << "xx_heads = ";
  for(const auto& data : xx_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "yy_heads = ";
  for(const auto& data : yy_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "zz_heads = ";
  for(const auto& data : zz_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  size_t block_size = NTX_MM_V5 * NTY_MM_V5 * NTZ_MM_V5;
  size_t grid_size;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V5; tt++) {

    // for each hyperplane, use one kernel
    for(size_t h=0; h<hyperplane_heads.size(); h++) {
      grid_size = xx_num * yy_num * zz_num * hyperplane_sizes[h];
      std::cout << "for hyperplane " << h << ", grid_size = " << grid_size << "\n";
      _updateEH_mix_mapping_ver5(Ex_pad_src, Ey_pad_src, Ez_pad_src,
                                 Hx_pad_src, Hy_pad_src, Hz_pad_src,
                                 Ex_pad_temp, Ey_pad_temp, Ez_pad_temp,
                                 Hx_pad_temp, Hy_pad_temp, Hz_pad_temp,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx,
                                 _Nx, _Ny, _Nz,
                                 Nx_pad, Ny_pad, Nz_pad,
                                 xx_num, yy_num, zz_num,
                                 xx_heads,
                                 yy_heads,
                                 zz_heads,
                                 hyperplane_sizes[h],
                                 hyperplane_heads[h],
                                 hyperplanes,
                                 block_size,
                                 grid_size);
    }

  }


}   

} // end of namespace gdiamond

#endif


















