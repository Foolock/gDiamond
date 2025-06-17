#ifndef GDIAMOND_GPU_MM_VER5_LS_CUH
#define GDIAMOND_GPU_MM_VER5_LS_CUH

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

void gDiamond::_updateEH_mix_mapping_ver5_ls(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                             std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                             std::vector<float>& Ex_pad_rep, std::vector<float>& Ey_pad_rep, std::vector<float>& Ez_pad_rep,
                                             std::vector<float>& Hx_pad_rep, std::vector<float>& Hy_pad_rep, std::vector<float>& Hz_pad_rep,
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
                                             size_t& count,
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
    
    /*
     * ---------------------
     * map a subtile within a supertile to a block
     * ---------------------
     */

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

    // Convert sub_xx == 0 into a binary flag
    const int is_rep_x = (sub_xx == 0); // 1 if replication, 0 otherwise
    const int is_rep_y = (sub_yy == 0);
    const int is_rep_z = (sub_zz == 0);

    // length of this subtile
    const int sub_xx_len = is_rep_x * (BLX_R + 1) + (1 - is_rep_x) * (BLX_P + 4);
    const int sub_yy_len = is_rep_y * (BLY_R + 1) + (1 - is_rep_y) * (BLY_P + 4);
    const int sub_zz_len = is_rep_z * (BLZ_R + 1) + (1 - is_rep_z) * (BLZ_P + 4);

    /*
     * ---------------------
     * decide the head indices of each subtile 
     * ---------------------
     */

    const int xx_head = xx_heads[super_xx] + (sub_xx > 0) * (BLX_R + (sub_xx - 1) * BLX_P);
    const int yy_head = yy_heads[super_yy] + (sub_yy > 0) * (BLY_R + (sub_yy - 1) * BLY_P);
    const int zz_head = zz_heads[super_zz] + (sub_zz > 0) * (BLZ_R + (sub_zz - 1) * BLZ_P);

    // if(super_xx == 0 && super_yy == 0 && super_zz == 0) {
    //   std::cout << "sub_xx = " << sub_xx 
    //             << ", sub_yy = " << sub_yy 
    //             << ", sub_zz = " << sub_zz
    //             << ", xx_head = " << xx_head
    //             << ", yy_head = " << yy_head
    //             << ", zz_head = " << zz_head
    //             << "\n";
    // }

    /*
     * ---------------------
     * declare shared memory 
     * ---------------------
     */

    // parallelogram calculation used more shared memory than replication calculation
    float shared_mem[H_SHX_V5 * H_SHY_V5 * H_SHZ_V5 * 3 + E_SHX_V5 * E_SHY_V5 * E_SHZ_V5 * 3];  
    constexpr int H_size = H_SHX_V5 * H_SHY_V5 * H_SHZ_V5;
    constexpr int E_size = E_SHX_V5 * E_SHY_V5 * E_SHZ_V5;
    float* Hx_shmem = shared_mem;
    float* Hy_shmem = Hx_shmem + H_size;
    float* Hz_shmem = Hy_shmem + H_size;
    float* Ex_shmem = Hz_shmem + H_size;
    float* Ey_shmem = Ex_shmem + E_size;
    float* Ez_shmem = Ey_shmem + E_size;

    /*
     * ---------------------
     * load shared memory   
     * ---------------------
     */

    const int H_halo_range = 3;  // H halo load range: 0,1,2
    const int E_halo_range = 4;  

    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V5;
      local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
      local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

      /*
       * ---------------------
       * load shared memory, load H   
       * ---------------------
       */

      for(H_shared_z = local_z; H_shared_z <= sub_zz_len - 1; H_shared_z += NTZ_MM_V5) {
        for(H_shared_y = local_y; H_shared_y <= sub_yy_len - 1; H_shared_y += NTY_MM_V5) {
          for(H_shared_x = local_x; H_shared_x <= sub_xx_len - 1; H_shared_x += NTX_MM_V5) {

            // set global index by H_shared index
            // If loading replication (sub == 0), global = head - 1 + H_shared
            // If loading parallelogram (sub > 0), global = head - 4 + H_shared
            global_x = xx_head - 1 + H_shared_x - 3 * (sub_xx > 0);
            global_y = yy_head - 1 + H_shared_y - 3 * (sub_yy > 0);
            global_z = zz_head - 1 + H_shared_z - 3 * (sub_zz > 0);

            H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;
            global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

            /*
             * decide where to load H, rep or src
             */
 
            float* Hx_pad_load;
            float* Hy_pad_load;
            float* Hz_pad_load;
          
            const int H_halo_mask = (H_shared_x < H_halo_range) << 2 | 
                                    (H_shared_y < H_halo_range) << 1 |
                                    (H_shared_z < H_halo_range);

            switch (H_halo_mask) {
              case 0b000:  // src area for H 
                Hx_pad_load = Hx_pad_src.data();
                Hy_pad_load = Hy_pad_src.data();
                Hz_pad_load = Hz_pad_src.data();
              case 0b001:  // z
                // load from (x, y, z-1)
                Hx_pad_load = (sub_zz - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_zz - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_zz - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b010:  // y
                // load from (x, y-1, z)
                Hx_pad_load = (sub_yy - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_yy - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_yy - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b011:  // y & z
                // load from (x, y-1, z-1)
                Hx_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b100:  // x
                // load from (x-1, y, z)
                Hx_pad_load = (sub_xx - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_xx - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_xx - 1 == 0)? Hy_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b101:  // x & z
                // load from (x-1, y, z-1)
                Hx_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b110:  // x & y
                // load from (x-1, y-1, z)
                Hx_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              case 0b111:  // x & y & z
                // load from (x-1, y-1, z-1)
                Hx_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hx_pad_rep.data() : Hx_pad_src.data();
                Hy_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hy_pad_rep.data() : Hy_pad_src.data();
                Hz_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Hz_pad_rep.data() : Hz_pad_src.data();
                break;
              default:
                break; // shouldn't reach here
            }

            // if(super_xx == 0 && super_yy == 0 && super_zz == 0) {
            //   std::cout << "------------------------------------------------------\n";
            //   if(loadH_from_rep) std::cout << "load from rep, ";
            //   else std::cout << "load from src, ";
            //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
            //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
            //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
            // }

            Hx_shmem[H_shared_idx] = Hx_pad_load[global_idx];
            Hy_shmem[H_shared_idx] = Hy_pad_load[global_idx];
            Hz_shmem[H_shared_idx] = Hz_pad_load[global_idx];
          }
        }
      } 

      /*
       * ---------------------
       * load shared memory, load E   
       * ---------------------
       */

      for(E_shared_z = local_z; E_shared_z <= sub_zz_len - 1; E_shared_z += NTZ_MM_V5) {
        for(E_shared_y = local_y; E_shared_y <= sub_yy_len - 1; E_shared_y += NTY_MM_V5) {
          for(E_shared_x = local_x; E_shared_x <= sub_xx_len - 1; E_shared_x += NTX_MM_V5) {

            // set global index by E_shared index
            // If loading replication (sub == 0), global = head - 1 + E_shared
            // If loading parallelogram (sub > 0), global = head - 4 + E_shared
            global_x = xx_head - 1 + E_shared_x - 3 * (sub_xx > 0);
            global_y = yy_head - 1 + E_shared_y - 3 * (sub_yy > 0);
            global_z = zz_head - 1 + E_shared_z - 3 * (sub_zz > 0);

            E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;
            global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

            /*
             * decide where to load E, rep or src
             */
            
            float* Ex_pad_load;
            float* Ey_pad_load;
            float* Ez_pad_load;
 
            const int E_halo_mask = (E_shared_x < E_halo_range) << 2 | 
                                    (E_shared_y < E_halo_range) << 1 |
                                    (E_shared_z < E_halo_range);

            switch (E_halo_mask) {
              case 0b000:  // src area for E 
                Ex_pad_load = Ex_pad_src.data();
                Ey_pad_load = Ey_pad_src.data();
                Ez_pad_load = Ez_pad_src.data();
              case 0b001:  // z
                // load from (x, y, z-1)
                Ex_pad_load = (sub_zz - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_zz - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_zz - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();
                break;
              case 0b010:  // y
                // load from (x, y-1, z)
                Ex_pad_load = (sub_yy - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_yy - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_yy - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();

                // if(global_x == 9 && global_y == 15 && global_z == 18 &&
                //    (sub_xx == 1 && sub_yy == 2 && sub_zz == 2)) {
                // if(global_x == 9 && global_y == 15 && global_z == 18) {
                //   if(sub_yy - 1 == 0) std::cout << "load from rep\n";
                //   else std::cout << "load from src\n"; 
                //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
                //   std::cout << "Ey_pad_load[global_idx] = " << Ey_pad_load[global_idx] << "\n";
                // }

                break;
              case 0b011:  // y & z
                // load from (x, y-1, z-1)
                Ex_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();
                break;
              case 0b100:  // x
                // load from (x-1, y, z)
                Ex_pad_load = (sub_xx - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_xx - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_xx - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();
                break;
              case 0b101:  // x & z
                // load from (x-1, y, z-1)
                Ex_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_xx - 1 == 0 || sub_zz - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();
                break;
              case 0b110:  // x & y
                // load from (x-1, y-1, z)
                Ex_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();

                // if(global_x == 7 && global_y == 15 && global_z == 18) {
                //   if(sub_xx - 1 == 0 || sub_yy - 1 == 0) std::cout << "load from rep\n";
                //   else std::cout << "load from src\n"; 
                //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
                //   std::cout << "Ey_pad_load[global_idx] = " << Ey_pad_load[global_idx] << "\n";
                // }

                break;
              case 0b111:  // x & y & z
                // load from (x-1, y-1, z-1)
                Ex_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ex_pad_rep.data() : Ex_pad_src.data();
                Ey_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ey_pad_rep.data() : Ey_pad_src.data();
                Ez_pad_load = (sub_xx - 1 == 0 || sub_yy - 1 == 0 || sub_zz - 1 == 0)? Ez_pad_rep.data() : Ez_pad_src.data();
                break;
              default:
                break; // shouldn't reach here
            }

            // if(super_xx == 0 && super_yy == 0 && super_zz == 0) {
            //   std::cout << "------------------------------------------------------\n";
            //   if(loadE_from_rep) std::cout << "load from rep, ";
            //   else std::cout << "load from src, ";
            //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
            //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
            //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
            // }

            // if(global_x == 7 && global_y == 15 && global_z == 18 &&
            //    (sub_xx == 1 && sub_yy == 2 && sub_zz == 2)) {
            //   std::cout << "loading here\n";
            //   std::cout << "E_halo_mask = " << E_halo_mask << ", ";
            //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
            //   std::cout << "Ey_pad_load[global_idx] = " << Ey_pad_load[global_idx] << "\n";
            // }

            Ex_shmem[E_shared_idx] = Ex_pad_load[global_idx];
            Ey_shmem[E_shared_idx] = Ey_pad_load[global_idx];
            Ez_shmem[E_shared_idx] = Ez_pad_load[global_idx];
          }
        }
      }
    }

    /*
     * ---------------------
     * calculation   
     * ---------------------
     */

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
    // Signed direction: +1 if replication, -1 if parallelogram
    int sgn_x = 2 * is_rep_x - 1;  // +1 or -1
    int sgn_y = 2 * is_rep_y - 1;
    int sgn_z = 2 * is_rep_z - 1;

    for(int t = 0; t < BLT_MM_V5; t++) {

      /*
       * ---------------------
       * decide calculation bounds   
       * ---------------------
       */

      // if calculating replication (sub == 0), calculation bounds follow mountain tiling
      // if calculating parallelogram (sub > 0), calculation bounds follow parallelogram tiling
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
      if(super_xx == 0 && super_yy == 0 && super_zz == 0 && 
        (sub_xx == 0 && sub_yy == 2 && sub_zz == 2)) {
        std::cout << "t = " << t << "\n";
        std::cout << "calE_head_X = " << calE_head_X << ", calE_tail_X = " << calE_tail_X
                  << ", calH_head_X = " << calH_head_X << ", calH_tail_X = " << calH_tail_X << "\n";
        std::cout << "calE_head_Y = " << calE_head_Y << ", calE_tail_Y = " << calE_tail_Y
                  << ", calH_head_Y = " << calH_head_Y << ", calH_tail_Y = " << calH_tail_Y << "\n";
        std::cout << "calE_head_Z = " << calE_head_Z << ", calE_tail_Z = " << calE_tail_Z
                  << ", calH_head_Z = " << calH_head_Z << ", calH_tail_Z = " << calH_tail_Z << "\n";
      }

      /*
       * ---------------------
       * update E   
       * ---------------------
       */

      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V5;
        local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
        local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

        // set H_shared and E_shared index by local index 
        // If calculating E in replication (sub == 0), H_shared = local, E_shared = local 
        // If calculating E in parallelogram (sub > 0), H_shared = local + 4 - t, E_shared = local + 4 - t 
        H_shared_x = local_x + (sub_xx > 0) * (4 - t);
        H_shared_y = local_y + (sub_yy > 0) * (4 - t);
        H_shared_z = local_z + (sub_zz > 0) * (4 - t);
        E_shared_x = local_x + (sub_xx > 0) * (4 - t);
        E_shared_y = local_y + (sub_yy > 0) * (4 - t);
        E_shared_z = local_z + (sub_zz > 0) * (4 - t);

        // set global index by local index
        // if calculating E in replication (sub == 0), global = head + local - 1
        // if calculating E in parallelogram (sub > 0), global = head + local - t 
        global_x = local_x + xx_head - (sub_xx == 0) * 1 - (sub_xx > 0) * t;
        global_y = local_y + yy_head - (sub_yy == 0) * 1 - (sub_yy > 0) * t;
        global_z = local_z + zz_head - (sub_zz == 0) * 1 - (sub_zz > 0) * t;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        global_idx = (global_x - LEFT_PAD_MM_V5) + (global_y - LEFT_PAD_MM_V5) * Nx + (global_z - LEFT_PAD_MM_V5) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;

        if(global_x == 7 && global_y == 15 && global_z == 18 && 
           (sub_xx == 0 && sub_yy == 2 && sub_zz == 2)) {
          std::cout << "???????";
          std::cout << "t = " << t << "\n";
          std::cout << "calE_head_X = " << calE_head_X << ", calE_tail_X = " << calE_tail_X
                    << ", calH_head_X = " << calH_head_X << ", calH_tail_X = " << calH_tail_X << "\n";
          std::cout << "calE_head_Y = " << calE_head_Y << ", calE_tail_Y = " << calE_tail_Y
                    << ", calH_head_Y = " << calH_head_Y << ", calH_tail_Y = " << calH_tail_Y << "\n";
          std::cout << "calE_head_Z = " << calE_head_Z << ", calE_tail_Z = " << calE_tail_Z
                    << ", calH_head_Z = " << calH_head_Z << ", calH_tail_Z = " << calH_tail_Z << "\n";
        }

        if(global_x >= 1 + LEFT_PAD_MM_V5 && global_x <= Nx - 2 + LEFT_PAD_MM_V5 &&
           global_y >= 1 + LEFT_PAD_MM_V5 && global_y <= Ny - 2 + LEFT_PAD_MM_V5 &&
           global_z >= 1 + LEFT_PAD_MM_V5 && global_z <= Nz - 2 + LEFT_PAD_MM_V5 &&
           global_x >= calE_head_X && global_x <= calE_tail_X &&
           global_y >= calE_head_Y && global_y <= calE_tail_Y &&
           global_z >= calE_head_Z && global_z <= calE_tail_Z) {

          // if(global_x == 8 && global_y == 15 && global_z == 18) {
          //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
          //   std::cout << "t = " << t << ", ";
          //   std::cout << "initially, Ey_shmem[E_shared_idx] = " << Ey_shmem[E_shared_idx] << ", ";
          // }
          Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                    ((Hz_shmem[H_shared_idx] + Hz_shmem[H_shared_idx - H_SHX_V5]) + (Hy_shmem[H_shared_idx] + Hy_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5]) + Jx[global_idx] * dx);

          Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                    ((Hx_shmem[H_shared_idx] + Hx_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5]) + (Hz_shmem[H_shared_idx] + Hz_shmem[H_shared_idx - 1]) + Jy[global_idx] * dx);

          Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                    ((Hy_shmem[H_shared_idx] + Hy_shmem[H_shared_idx - 1]) + (Hx_shmem[H_shared_idx] + Hx_shmem[H_shared_idx - H_SHX_V5]) + Jz[global_idx] * dx);

          // if(super_xx == 1 && super_yy == 1 && super_zz == 0 && sub_xx == 0 && sub_yy == 0 && sub_zz == 2) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }

          // if(global_x == 7 && global_y == 15 && global_z == 18) {
          //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
          //   std::cout << "t = " << t << ", ";
          //   std::cout << "Ex_shmem[E_shared_idx] = " << Ex_shmem[E_shared_idx] << ", ";
          //   std::cout << "Hz_shmem[H_shared_idx] = " << Hz_shmem[H_shared_idx] << ", ";
          //   std::cout << "Hz_shmem[H_shared_idx - H_SHX_V5] = " << Hz_shmem[H_shared_idx - H_SHX_V5] << ", ";
          //   std::cout << "Hy_shmem[H_shared_idx] = " << Hy_shmem[H_shared_idx] << ", ";
          //   std::cout << "Hy_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5] = " << Hy_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5] << "\n";
          // }

          // if(global_x == 8 && global_y == 15 && global_z == 18) {
          if(global_x == 7 && global_y == 15 && global_z == 18 && 
             (sub_xx == 0 && sub_yy == 2 && sub_zz == 2)) {
            std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
            std::cout << "t = " << t << ", ";
            std::cout << "Ey_shmem[E_shared_idx] = " << Ey_shmem[E_shared_idx] << ", ";
            std::cout << "Hx_shmem[H_shared_idx] = " << Hx_shmem[H_shared_idx] << ", ";
            std::cout << "Hx_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5] = " << Hx_shmem[H_shared_idx - H_SHX_V5 * H_SHY_V5] << ", ";
            std::cout << "Hz_shmem[H_shared_idx] = " << Hz_shmem[H_shared_idx] << ", ";
            std::cout << "Hz_shmem[H_shared_idx - 1] = " << Hz_shmem[H_shared_idx - 1] << "\n";
          }
  
          count++;

        }
      }

      /*
       * ---------------------
       * update H   
       * ---------------------
       */

      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V5;
        local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
        local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

        // set H_shared and E_shared index by local index 
        // If calculating H in replication (sub == 0), H_shared = local, E_shared = local 
        // If calculating H in parallelogram (sub > 0), H_shared = local + 3 - t, E_shared = local + 3 - t 
        H_shared_x = local_x + (sub_xx > 0) * (3 - t);
        H_shared_y = local_y + (sub_yy > 0) * (3 - t);
        H_shared_z = local_z + (sub_zz > 0) * (3 - t);
        E_shared_x = local_x + (sub_xx > 0) * (3 - t);
        E_shared_y = local_y + (sub_yy > 0) * (3 - t);
        E_shared_z = local_z + (sub_zz > 0) * (3 - t);

        // set global index by local index
        // if calculating H in replication (sub == 0), global = head + local - 1
        // if calculating H in parallelogram (sub > 0), global = head + local - t - 1 
        global_x = local_x + xx_head - 1 - (sub_xx > 0) * t;
        global_y = local_y + yy_head - 1 - (sub_yy > 0) * t;
        global_z = local_z + zz_head - 1 - (sub_zz > 0) * t;

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
                    ((Ey_shmem[E_shared_idx + E_SHX_V5 * E_SHY_V5] + Ey_shmem[E_shared_idx]) + (Ez_shmem[E_shared_idx + E_SHX_V5] + Ez_shmem[E_shared_idx]) + Mx[global_idx] * dx);

          Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                    ((Ez_shmem[E_shared_idx + 1] + Ez_shmem[E_shared_idx]) + (Ex_shmem[E_shared_idx + E_SHX_V5 * E_SHY_V5] + Ex_shmem[E_shared_idx]) + My[global_idx] * dx);

          Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                    ((Ex_shmem[E_shared_idx + E_SHX_V5] + Ex_shmem[E_shared_idx]) + (Ey_shmem[E_shared_idx + 1] + Ey_shmem[E_shared_idx]) + Mz[global_idx] * dx);

          // if(super_xx == 0 && super_yy == 0 && super_zz == 0 && sub_xx == 1 && sub_yy == 1 && sub_zz == 2) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }

          // if(global_x == 7 && global_y == 15 && global_z == 18) {
          //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
          //   std::cout << "t = " << t << ", ";
          //   std::cout << "Hz_shmem[H_shared_idx] = " << Hz_shmem[H_shared_idx] << ", ";
          //   std::cout << "Ex_shmem[E_shared_idx + E_SHX_V5] = " << Ex_shmem[E_shared_idx + E_SHX_V5] << ", ";
          //   std::cout << "Ex_shmem[E_shared_idx] = " << Ex_shmem[E_shared_idx] << ", ";
          //   std::cout << "Ey_shmem[E_shared_idx + 1] = " << Ey_shmem[E_shared_idx + 1] << ", ";
          //   std::cout << "Ey_shmem[E_shared_idx] = " << Ey_shmem[E_shared_idx] << "\n";
          // }

        }
      }
    }

    /*
     * ---------------------
     * store to global memory   
     * ---------------------
     */

    // we only store to src when all dimensions are parallelograms (sub > 0)
    const bool store_to_src = (sub_xx > 0) && (sub_yy > 0) && (sub_zz > 0);
    float* Hx_pad_dst = (store_to_src)? Hx_pad_src.data() : Hx_pad_rep.data();
    float* Hy_pad_dst = (store_to_src)? Hy_pad_src.data() : Hy_pad_rep.data();
    float* Hz_pad_dst = (store_to_src)? Hz_pad_src.data() : Hz_pad_rep.data();
    float* Ex_pad_dst = (store_to_src)? Ex_pad_src.data() : Ex_pad_rep.data();
    float* Ey_pad_dst = (store_to_src)? Ey_pad_src.data() : Ey_pad_rep.data();
    float* Ez_pad_dst = (store_to_src)? Ez_pad_src.data() : Ez_pad_rep.data();

    /*
     * ---------------------
     * decide storing bounds   
     * ---------------------
     */

    // If storing replication (sub == 0), storeE_head = head + 3
    //                                    storeE_tail = storeE_head + 3
    //                                    storeH_head = storeE_head
    //                                    storeH_tail = storeH_head + 2
    // If storing parallelogram (sub > 0), storeE_head = head - 3
    //                                     storeE_tail = storeE_head + sub_len - 2
    //                                     storeH_head = storeE_head - 1
    //                                     storeH_tail = storeH_head + sub_len - 2

    int storeE_head_X = (sub_xx == 0)? xx_head + 3 : xx_head - 3; 
    int storeE_tail_X = (sub_xx == 0)? storeE_head_X + 3 : storeE_head_X + sub_xx_len - 2; 
    int storeH_head_X = (sub_xx == 0)? storeE_head_X : storeE_head_X - 1;
    int storeH_tail_X = (sub_xx == 0)? storeH_head_X + 2 : storeH_head_X + sub_xx_len - 2;

    int storeE_head_Y = (sub_yy == 0)? yy_head + 3 : yy_head - 3; 
    int storeE_tail_Y = (sub_yy == 0)? storeE_head_Y + 3 : storeE_head_Y + sub_yy_len - 2; 
    int storeH_head_Y = (sub_yy == 0)? storeE_head_Y : storeE_head_Y - 1;
    int storeH_tail_Y = (sub_yy == 0)? storeH_head_Y + 2 : storeH_head_Y + sub_yy_len - 2;

    int storeE_head_Z = (sub_zz == 0)? zz_head + 3 : zz_head - 3; 
    int storeE_tail_Z = (sub_zz == 0)? storeE_head_Z + 3 : storeE_head_Z + sub_zz_len - 2; 
    int storeH_head_Z = (sub_zz == 0)? storeE_head_Z : storeE_head_Z - 1;
    int storeH_tail_Z = (sub_zz == 0)? storeH_head_Z + 2 : storeH_head_Z + sub_zz_len - 2;

    // std::cout << "super_xx = " << super_xx << ", super_yy = " << super_yy << ", super_zz = " << super_zz << "\n";
    // std::cout << "storeE_head_X = " << storeE_head_X << ", storeE_tail_X = " << storeE_tail_X
    //           << ", storeH_head_X = " << storeH_head_X << ", storeH_tail_X = " << storeH_tail_X << "\n";
    // std::cout << "storeE_head_Y = " << storeE_head_Y << ", storeE_tail_Y = " << storeE_tail_Y
    //           << ", storeH_head_Y = " << storeH_head_Y << ", storeH_tail_Y = " << storeH_tail_Y << "\n";
    // std::cout << "storeE_head_Z = " << storeE_head_Z << ", storeE_tail_Z = " << storeE_tail_Z
    //           << ", storeH_head_Z = " << storeH_head_Z << ", storeH_tail_Z = " << storeH_tail_Z << "\n";
    // std::cout << "\n";

    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V5;
      local_y = (thread_id / NTX_MM_V5) % NTY_MM_V5;
      local_z = thread_id / (NTX_MM_V5 * NTY_MM_V5);

      /*
       * ---------------------
       * store to global memory, store H 
       * ---------------------
       */

      for(H_shared_z = local_z; H_shared_z <= sub_zz_len - 1; H_shared_z += NTZ_MM_V5) {
        for(H_shared_y = local_y; H_shared_y <= sub_yy_len - 1; H_shared_y += NTY_MM_V5) {
          for(H_shared_x = local_x; H_shared_x <= sub_xx_len - 1; H_shared_x += NTX_MM_V5) {

            // set global index by H_shared index
            // If storing replication (sub == 0), global = head - 1 + H_shared
            // If storing parallelogram (sub > 0), global = head - 4 + H_shared
            global_x = xx_head - 1 + H_shared_x - 3 * (sub_xx > 0);
            global_y = yy_head - 1 + H_shared_y - 3 * (sub_yy > 0);
            global_z = zz_head - 1 + H_shared_z - 3 * (sub_zz > 0);

            H_shared_idx = H_shared_x + H_shared_y * H_SHX_V5 + H_shared_z * H_SHX_V5 * H_SHY_V5;
            global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

            // if(global_x >= 1 + LEFT_PAD_MM_V5 && global_x <= Nx - 2 + LEFT_PAD_MM_V5 &&
            //    global_y >= 1 + LEFT_PAD_MM_V5 && global_y <= Ny - 2 + LEFT_PAD_MM_V5 &&
            //    global_z >= 1 + LEFT_PAD_MM_V5 && global_z <= Nz - 2 + LEFT_PAD_MM_V5 &&
            //    global_x >= storeH_head_X && global_x <= storeH_tail_X &&
            //    global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
            //    global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {
            if(global_x >= storeH_head_X && global_x <= storeH_tail_X &&
               global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
               global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

              Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
              Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
              Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
            }
          }
        }
      }

      /*
       * ---------------------
       * store to global memory, store E 
       * ---------------------
       */

      for(E_shared_z = local_z; E_shared_z <= sub_zz_len - 1; E_shared_z += NTZ_MM_V5) {
        for(E_shared_y = local_y; E_shared_y <= sub_yy_len - 1; E_shared_y += NTY_MM_V5) {
          for(E_shared_x = local_x; E_shared_x <= sub_xx_len - 1; E_shared_x += NTX_MM_V5) {

            // set global index by E_shared index
            // If storing replication (sub == 0), global = head - 1 + E_shared
            // If storing parallelogram (sub > 0), global = head - 4 + E_shared
            global_x = xx_head - 1 + E_shared_x - 3 * (sub_xx > 0);
            global_y = yy_head - 1 + E_shared_y - 3 * (sub_yy > 0);
            global_z = zz_head - 1 + E_shared_z - 3 * (sub_zz > 0);

            E_shared_idx = E_shared_x + E_shared_y * E_SHX_V5 + E_shared_z * E_SHX_V5 * E_SHY_V5;  
            global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

            // if(global_x >= 1 + LEFT_PAD_MM_V5 && global_x <= Nx - 2 + LEFT_PAD_MM_V5 &&
            //    global_y >= 1 + LEFT_PAD_MM_V5 && global_y <= Ny - 2 + LEFT_PAD_MM_V5 &&
            //    global_z >= 1 + LEFT_PAD_MM_V5 && global_z <= Nz - 2 + LEFT_PAD_MM_V5 &&
            //    global_x >= storeE_head_X && global_x <= storeE_tail_X &&
            //    global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
            //    global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {
            if(global_x >= storeE_head_X && global_x <= storeE_tail_X &&
               global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
               global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

              // if(global_x == 8 && global_y == 15 && global_z == 18 &&
              //    (sub_xx == 1 && sub_yy == 1 && sub_zz == 2)) {
              //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
              //   if(store_to_src) std::cout << "store to src, ";
              //   else std::cout << "store to rep, ";
              //   std::cout << "Ey_shmem[E_shared_idx] = " << Ey_shmem[E_shared_idx] << "\n";
              // }

              // if(global_x == 9 && global_y == 15 && global_z == 18 &&
              //    (sub_xx == 1 && sub_yy == 1 && sub_zz == 2)) {
              
              Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
              Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
              Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];

              // if(global_x == 7 && global_y == 15 && global_z == 18) {
              //   if(store_to_src) std::cout << "storing in src\n";
              //   else std::cout << "storing in rep\n";
              //   std::cout << "sub_xx = " << sub_xx << ", sub_yy = " << sub_yy << ", sub_zz = " << sub_zz << "\n";
              //   std::cout << "Ey_shmem[E_shared_idx] = " << Ey_shmem[E_shared_idx] << "\n";
              //   std::cout << "Ey_pad_dst[global_idx] = " << Ey_pad_dst[global_idx] << "\n";
              // }

            }
          }
        }
      }
    }
  }

}



void gDiamond::update_FDTD_mix_mapping_sequential_ver5_larger_shmem(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  std::cout << "running update_FDTD_mix_mapping_sequential_ver5...\n";
  std::cout << "supertiles are the larger overlapped mountain tiles. " 
               "Within one supertile there are a bunch of subtiles, "
               "containing one small mountain and a series of parallelograms.\n";

  size_t count = 0;

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
  std::vector<float> Ex_pad_rep(padded_length, 0);
  std::vector<float> Ey_pad_rep(padded_length, 0);
  std::vector<float> Ez_pad_rep(padded_length, 0);
  std::vector<float> Hx_pad_rep(padded_length, 0);
  std::vector<float> Hy_pad_rep(padded_length, 0);
  std::vector<float> Hz_pad_rep(padded_length, 0);

  // fill src with 1s to check
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V5;
        size_t y_pad = y + LEFT_PAD_MM_V5;
        size_t z_pad = z + LEFT_PAD_MM_V5;
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        Ex_pad_src[padded_index] = 1;
        Ey_pad_src[padded_index] = 1;
        Ez_pad_src[padded_index] = 1;
        Hx_pad_src[padded_index] = 1;
        Hy_pad_src[padded_index] = 1;
        Hz_pad_src[padded_index] = 1;
      }
    }
  }

  for(size_t z = 0; z < Nz_pad; z++) {
    for(size_t y = 0; y < Ny_pad; y++) {
      for(size_t x = 0; x < Nx_pad; x++) {
        if(x == 7 && y == 19 && z == 18) {
          size_t padded_index = x + y * Nx_pad + z * Nx_pad * Ny_pad;
          std::cout << "Ex_pad_src[padded_index] = " << Ex_pad_src[padded_index] << "\n";
        }
      }
    }
  }

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

  // std::cout << "xx_heads = ";
  // for(const auto& data : xx_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads = ";
  // for(const auto& data : yy_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads = ";
  // for(const auto& data : zz_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";

  size_t block_size = NTX_MM_V5 * NTY_MM_V5 * NTZ_MM_V5;
  size_t grid_size;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V5; tt++) {

    // for each hyperplane, use one kernel
    for(size_t h=0; h<hyperplane_heads.size(); h++) {
      grid_size = xx_num * yy_num * zz_num * hyperplane_sizes[h];
      std::cout << "for hyperplane " << h << ", grid_size = " << grid_size << "\n";
      _updateEH_mix_mapping_ver5_ls(Ex_pad_src, Ey_pad_src, Ez_pad_src,
                                    Hx_pad_src, Hy_pad_src, Hz_pad_src,
                                    Ex_pad_rep, Ey_pad_rep, Ez_pad_rep,
                                    Hx_pad_rep, Hy_pad_rep, Hz_pad_rep,
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
                                    count,
                                    block_size,
                                    grid_size);

      // int check_index = 9 + 15 * Nx_pad + 18 * Nx_pad * Ny_pad;
      // std::cout << "-------------Ey_pad_src[check_index] = " << Ey_pad_src[check_index] << "\n";

    }
  }

  std::cout << "(_Nx - 2) * (_Ny - 2) * (_Nz - 2) * 4 = " << (_Nx - 2) * (_Ny - 2) * (_Nz - 2) * 4 << "\n"; 
  std::cout << "count = " << count << "\n";

  // transfer data back to unpadded arrays
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V5;
        size_t y_pad = y + LEFT_PAD_MM_V5;
        size_t z_pad = z + LEFT_PAD_MM_V5;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        _Ex_simu[unpadded_index] = Ex_pad_src[padded_index];
        _Ey_simu[unpadded_index] = Ey_pad_src[padded_index];
        _Ez_simu[unpadded_index] = Ez_pad_src[padded_index];
        _Hx_simu[unpadded_index] = Hx_pad_src[padded_index];
        _Hy_simu[unpadded_index] = Hy_pad_src[padded_index];
        _Hz_simu[unpadded_index] = Hz_pad_src[padded_index];
      }
    }
  }

}   

} // end of namespace gdiamond

#endif


















