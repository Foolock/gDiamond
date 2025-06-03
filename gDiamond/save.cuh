// store copy of implementations

__global__ void updateEH_mix_mapping_kernel_ver4_unroll(float* Ex_pad_src, float* Ey_pad_src, float* Ez_pad_src,
                                                 float* Hx_pad_src, float* Hy_pad_src, float* Hz_pad_src,
                                                 float* Ex_pad_rep, float* Ey_pad_rep, float* Ez_pad_rep,
                                                 float* Hx_pad_rep, float* Hy_pad_rep, float* Hz_pad_rep,
                                                 float* Ex_pad_dst, float* Ey_pad_dst, float* Ez_pad_dst,
                                                 float* Hx_pad_dst, float* Hy_pad_dst, float* Hz_pad_dst,
                                                 float* Cax, float* Cbx,
                                                 float* Cay, float* Cby,
                                                 float* Caz, float* Cbz,
                                                 float* Dax, float* Dbx,
                                                 float* Day, float* Dby,
                                                 float* Daz, float* Dbz,
                                                 float* Jx, float* Jy, float* Jz,
                                                 float* Mx, float* My, float* Mz,
                                                 float dx,
                                                 int Nx, int Ny, int Nz,
                                                 int Nx_pad, int Ny_pad, int Nz_pad,
                                                 int xx_num, int yy_num, int zz_num,
                                                 int* xx_heads,
                                                 int* yy_heads,
                                                 int* zz_heads) {
  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  const int xx = block_id % xx_num;
  const int yy = (block_id / xx_num) % yy_num;
  const int zz = block_id / (xx_num * yy_num);

  const int local_x = thread_id % NTX_MM_V4;
  const int local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
  const int local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);

  const int global_x = xx_heads[xx] + local_x;
  const int global_y = yy_heads[yy] + local_y;
  const int global_z = zz_heads[zz] + local_z;

  const int H_shared_x = local_x + 1;
  const int H_shared_y = local_y + 1;
  const int H_shared_z = local_z + 1;

  const int E_shared_x = local_x;
  const int E_shared_y = local_y;
  const int E_shared_z = local_z;

  int global_idx;
  int H_shared_idx;
  int E_shared_idx;

  // declare shared memory
  // parallelogram calculation used more shared memory than replication calculation
  __shared__ float Hx_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hy_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hz_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Ex_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ey_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ez_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];

  // load shared memory (replication)

  // load core ---------------------------------------------
  const int load_head_X = xx_heads[xx];
  const int load_tail_X = xx_heads[xx] + BLX_R - 1;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  if(global_x >= load_head_X && global_x <= load_tail_X) {
    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
    Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
  }

  // H HALO
  // E HALO is not needed since there is no valley tile in mix mapping ver4
  if (local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }

  __syncthreads();

  // calculation (replication)

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  unroll_cal_rep_loop<0>(
  global_idx, H_shared_idx, E_shared_idx,
  global_x, global_y, global_z,
  Nx, Ny, Nz,
  xx_heads, yy_heads, zz_heads,
  xx, yy, zz,
  dx,
  Cax, Cbx,
  Cay, Cby,
  Caz, Cbz,
  Dax, Dbx,
  Day, Dby,
  Daz, Dbz,
  Jx, Jy, Jz,
  Mx, My, Mz,
  Ex_shmem, Ey_shmem, Ez_shmem,
  Hx_shmem, Hy_shmem, Hz_shmem
  );

  // store back to global memory (replication)

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  const int storeE_head_X = xx_heads[xx] + BLX_R - BLT_MM_V4;
  const int storeE_tail_X = xx_heads[xx] + BLX_R - 1;
  const int storeH_head_X = storeE_head_X;
  const int storeH_tail_X = storeE_tail_X - 1;

  const int storeE_head_Y = yy_heads[yy] + BLY_R - BLT_MM_V4;
  const int storeE_tail_Y = yy_heads[yy] + BLY_R - 1;
  const int storeH_head_Y = storeE_head_Y;
  const int storeH_tail_Y = storeE_tail_Y - 1;

  const int storeE_head_Z = zz_heads[zz] + BLZ_R - BLT_MM_V4;
  const int storeE_tail_Z = zz_heads[zz] + BLZ_R - 1;
  const int storeH_head_Z = storeE_head_Z;
  const int storeH_tail_Z = storeE_tail_Z - 1;

  // store H ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_rep[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_rep[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_rep[global_idx] = Hz_shmem[H_shared_idx];
  }

  // store E ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_rep[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_rep[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_rep[global_idx] = Ez_shmem[E_shared_idx];
  }
}

void gDiamond::_updateEH_mix_mapping_ver4(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
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
                                          size_t grid_size) {

  for(size_t block_id = 0; block_id < grid_size; block_id++) {
    const int xx = block_id % xx_num;
    const int yy = (block_id / xx_num) % yy_num;
    const int zz = block_id / (xx_num * yy_num);
    int local_x, local_y, local_z;
    int global_x, global_y, global_z;
    int H_shared_x, H_shared_y, H_shared_z;
    int E_shared_x, E_shared_y, E_shared_z;
    int global_idx;
    int H_shared_idx;
    int E_shared_idx;

    // declare shared memory
    // parallelogram calculation used more shared memory than replication calculation
    float Hx_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Hy_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Hz_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Ex_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
    float Ey_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
    float Ez_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];

    // load shared memory (replication part)
    // since in X dimension, BLX_R = 8 and we are using 16 threads
    // we don't need to load that much of elements
    // these bounds are for loading core
    // so there is no difference between loadE and loadH
    // there is no explicit bound for loading in Y, Z dimension
    // bounds are refer to padded global_x
    const int load_head_X = xx_heads[xx];
    const int load_tail_X = xx_heads[xx] + BLX_R - 1;
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V4;
      local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
      local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);
      H_shared_x = local_x + 1;
      H_shared_y = local_y + 1;
      H_shared_z = local_z + 1;
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + local_x;
      global_y = yy_heads[yy] + local_y;
      global_z = zz_heads[zz] + local_z;

      // load core ---------------------------------------------
      H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
      E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
      global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      if(global_x >= load_head_X && global_x <= load_tail_X) {
        Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
        Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
      }

      // H HALO
      // E HALO is not needed since there is no valley tile in mix mapping ver4
      if (local_x == 0) {
        int halo_x = 0;
        int global_x_halo = xx_heads[xx] + halo_x - 1;

        global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

        Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
      }
      if (local_y == 0) {
        int halo_y = 0;
        int global_y_halo = yy_heads[yy] + halo_y - 1;

        global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

        Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
      }
      if (local_z == 0) {
        int halo_z = 0;
        int global_z_halo = zz_heads[zz] + halo_z - 1;

        global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

        Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
      }
    }

    // calculation (replication)
    for(int t = 0; t < BLT_MM_V4; t++) {

      // X head and tail is refer to padded global_x
      // same thing applys to Y and Z
      int calE_head_X = xx_heads[xx] + t;
      int calE_tail_X = xx_heads[xx] + BLX_R - 1 - t;
      int calH_head_X = calE_head_X;
      int calH_tail_X = calE_tail_X - 1;

      int calE_head_Y = yy_heads[yy] + t;
      int calE_tail_Y = yy_heads[yy] + BLY_R - 1 - t;
      int calH_head_Y = calE_head_Y;
      int calH_tail_Y = calE_tail_Y - 1;

      int calE_head_Z = zz_heads[zz] + t;
      int calE_tail_Z = zz_heads[zz] + BLZ_R - 1 - t;
      int calH_head_Z = calE_head_Z;
      int calH_tail_Z = calE_tail_Z - 1;

      // if(xx == 0 && yy == 0 && zz == 0) {
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
        local_x = thread_id % NTX_MM_V4;
        local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
        local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);
        H_shared_x = local_x + 1;
        H_shared_y = local_y + 1;
        H_shared_z = local_z + 1;
        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

        if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
           global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
           global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
           global_x >= calE_head_X && global_x <= calE_tail_X &&
           global_y >= calE_head_Y && global_y <= calE_tail_Y &&
           global_z >= calE_head_Z && global_z <= calE_tail_Z) {

          Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                    ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V4]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - Jx[global_idx] * dx);

          Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                    ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

          Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                    ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4]) - Jz[global_idx] * dx);
        }
      }

      // update H
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V4;
        local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
        local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);
        H_shared_x = local_x + 1;
        H_shared_y = local_y + 1;
        H_shared_z = local_z + 1;
        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

        if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
           global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
           global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
           global_x >= calH_head_X && global_x <= calH_tail_X &&
           global_y >= calH_head_Y && global_y <= calH_tail_Y &&
           global_z >= calH_head_Z && global_z <= calH_tail_Z) {

          Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                    ((Ey_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V4] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

          Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                    ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

          Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                    ((Ex_shmem[E_shared_idx + E_SHX_V4] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
        }
      }
    }

    // store back to global memory (replication)

    // X head and tail is refer to padded global_x
    // same thing applys to Y and Z
    const int storeE_head_X = xx_heads[xx] + BLX_R - BLT_MM_V4;
    const int storeE_tail_X = xx_heads[xx] + BLX_R - 1;
    const int storeH_head_X = storeE_head_X;
    const int storeH_tail_X = storeE_tail_X - 1;

    const int storeE_head_Y = yy_heads[yy] + BLY_R - BLT_MM_V4;
    const int storeE_tail_Y = yy_heads[yy] + BLY_R - 1;
    const int storeH_head_Y = storeE_head_Y;
    const int storeH_tail_Y = storeE_tail_Y - 1;

    const int storeE_head_Z = zz_heads[zz] + BLZ_R - BLT_MM_V4;
    const int storeE_tail_Z = zz_heads[zz] + BLZ_R - 1;
    const int storeH_head_Z = storeE_head_Z;
    const int storeH_tail_Z = storeE_tail_Z - 1;

    // if(xx == 0 && yy == 0 && zz == 0) {
    //   std::cout << "storeE_head_X = " << storeE_head_X << ", storeE_tail_X = " << storeE_tail_X
    //             << ", storeH_head_X = " << storeH_head_X << ", storeH_tail_X = " << storeH_tail_X << "\n";
    //   std::cout << "storeE_head_Y = " << storeE_head_Y << ", storeE_tail_Y = " << storeE_tail_Y
    //             << ", storeH_head_Y = " << storeH_head_Y << ", storeH_tail_Y = " << storeH_tail_Y << "\n";
    //   std::cout << "storeE_head_Z = " << storeE_head_Z << ", storeE_tail_Z = " << storeE_tail_Z
    //             << ", storeH_head_Z = " << storeH_head_Z << ", storeH_tail_Z = " << storeH_tail_Z << "\n";
    // }

    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V4;
      local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
      local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);
      H_shared_x = local_x + 1;
      H_shared_y = local_y + 1;
      H_shared_z = local_z + 1;
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + E_shared_x;
      global_y = yy_heads[yy] + E_shared_y;
      global_z = zz_heads[zz] + E_shared_z;

      // store H ---------------------------------------------
      if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
         global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
         global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
         global_x >= storeH_head_X && global_x <= storeH_tail_X &&
         global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
         global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
        Hx_pad_rep[global_idx] = Hx_shmem[H_shared_idx];
        Hy_pad_rep[global_idx] = Hy_shmem[H_shared_idx];
        Hz_pad_rep[global_idx] = Hz_shmem[H_shared_idx];
      }

      // store E ---------------------------------------------
      if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
         global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
         global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
         global_x >= storeE_head_X && global_x <= storeE_tail_X &&
         global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
         global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
        Ex_pad_rep[global_idx] = Ex_shmem[E_shared_idx];
        Ey_pad_rep[global_idx] = Ey_shmem[E_shared_idx];
        Ez_pad_rep[global_idx] = Ez_shmem[E_shared_idx];
      }
    }
  }
}

void gDiamond::update_FDTD_gpu_simulation_check(size_t num_timesteps) {

  std::cout << "running update_FDTD_gpu_simulation and update_FDTD_gpu_simulation_shmem_EH\n"; 

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  size_t total_cal = 0;
  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
    
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_mountains_X, num_mountains_Y, num_mountains_Z, 
                                 mountain_heads_X, mountain_heads_Y, mountain_heads_Z, 
                                 mountain_tails_X, mountain_tails_Y, mountain_tails_Z, 
                                 1, 1, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_mountains_X, num_mountains_Y, num_mountains_Z, 
                        mountain_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        1, 1, 1,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 1. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_valleys_X, num_mountains_Y, num_mountains_Z, 
                                 valley_heads_X, mountain_heads_Y, mountain_heads_Z, 
                                 valley_tails_X, mountain_tails_Y, mountain_tails_Z, 
                                 0, 1, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_valleys_X, num_mountains_Y, num_mountains_Z, 
                        valley_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        valley_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        0, 1, 1,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 2. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_mountains_X, num_valleys_Y, num_mountains_Z, 
                                 mountain_heads_X, valley_heads_Y, mountain_heads_Z, 
                                 mountain_tails_X, valley_tails_Y, mountain_tails_Z, 
                                 1, 0, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_mountains_X, num_valleys_Y, num_mountains_Z, 
                        mountain_heads_X, valley_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, valley_tails_Y, mountain_tails_Z, 
                        1, 0, 1,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 3. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_mountains_X, num_mountains_Y, num_valleys_Z, 
                                 mountain_heads_X, mountain_heads_Y, valley_heads_Z, 
                                 mountain_tails_X, mountain_tails_Y, valley_tails_Z, 
                                 1, 1, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_mountains_X, num_mountains_Y, num_valleys_Z, 
                        mountain_heads_X, mountain_heads_Y, valley_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, valley_tails_Z, 
                        1, 1, 0,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 4. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_valleys_X, num_valleys_Y, num_mountains_Z, 
                                 valley_heads_X, valley_heads_Y, mountain_heads_Z, 
                                 valley_tails_X, valley_tails_Y, mountain_tails_Z, 
                                 0, 0, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_valleys_X, num_valleys_Y, num_mountains_Z, 
                        valley_heads_X, valley_heads_Y, mountain_heads_Z, 
                        valley_tails_X, valley_tails_Y, mountain_tails_Z, 
                        0, 0, 1,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 5. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_valleys_X, num_mountains_Y, num_valleys_Z, 
                                 valley_heads_X, mountain_heads_Y, valley_heads_Z, 
                                 valley_tails_X, mountain_tails_Y, valley_tails_Z, 
                                 0, 1, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_valleys_X, num_mountains_Y, num_valleys_Z, 
                        valley_heads_X, mountain_heads_Y, valley_heads_Z, 
                        valley_tails_X, mountain_tails_Y, valley_tails_Z, 
                        0, 1, 0,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 6. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_mountains_X, num_valleys_Y, num_valleys_Z, 
                                 mountain_heads_X, valley_heads_Y, valley_heads_Z, 
                                 mountain_tails_X, valley_tails_Y, valley_tails_Z, 
                                 1, 0, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_mountains_X, num_valleys_Y, num_valleys_Z, 
                        mountain_heads_X, valley_heads_Y, valley_heads_Z, 
                        mountain_tails_X, valley_tails_Y, valley_tails_Z, 
                        1, 0, 0,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 7. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
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
                                 num_valleys_X, num_valleys_Y, num_valleys_Z, 
                                 valley_heads_X, valley_heads_Y, valley_heads_Z, 
                                 valley_tails_X, valley_tails_Y, valley_tails_Z, 
                                 0, 0, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
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
                        num_valleys_X, num_valleys_Y, num_valleys_Z, 
                        valley_heads_X, valley_heads_Y, valley_heads_Z, 
                        valley_tails_X, valley_tails_Y, valley_tails_Z, 
                        0, 0, 0,
                        block_size,
                        grid_size);

    if(!check_correctness_simu_shmem()) {
      std::cout << "phase 8. t = " << t << "\n";
      std::cerr << "error: results not match\n";
      std::exit(EXIT_FAILURE);
    }
  }

}


void gDiamond::update_FDTD_gpu_simulation(size_t num_timesteps) { // simulation of gpu threads

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

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
    
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_mountains_X, num_mountains_Y, num_mountains_Z, 
                        mountain_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        1, 1, 1,
                        block_size,
                        grid_size);

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_valleys_X, num_mountains_Y, num_mountains_Z, 
                        valley_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        valley_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        0, 1, 1,
                        block_size,
                        grid_size);

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_mountains_X, num_valleys_Y, num_mountains_Z, 
                        mountain_heads_X, valley_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, valley_tails_Y, mountain_tails_Z, 
                        1, 0, 1,
                        block_size,
                        grid_size);

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_mountains_X, num_mountains_Y, num_valleys_Z, 
                        mountain_heads_X, mountain_heads_Y, valley_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, valley_tails_Z, 
                        1, 1, 0,
                        block_size,
                        grid_size);

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_valleys_X, num_valleys_Y, num_mountains_Z, 
                        valley_heads_X, valley_heads_Y, mountain_heads_Z, 
                        valley_tails_X, valley_tails_Y, mountain_tails_Z, 
                        0, 0, 1,
                        block_size,
                        grid_size);

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_valleys_X, num_mountains_Y, num_valleys_Z, 
                        valley_heads_X, mountain_heads_Y, valley_heads_Z, 
                        valley_tails_X, mountain_tails_Y, valley_tails_Z, 
                        0, 1, 0,
                        block_size,
                        grid_size);

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_mountains_X, num_valleys_Y, num_valleys_Z, 
                        mountain_heads_X, valley_heads_Y, valley_heads_Z, 
                        mountain_tails_X, valley_tails_Y, valley_tails_Z, 
                        1, 0, 0,
                        block_size,
                        grid_size);

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq(Ex_temp, Ey_temp, Ez_temp,
                        Hx_temp, Hy_temp, Hz_temp,
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
                        num_valleys_X, num_valleys_Y, num_valleys_Z, 
                        valley_heads_X, valley_heads_Y, valley_heads_Z, 
                        valley_tails_X, valley_tails_Y, valley_tails_Z, 
                        0, 0, 0,
                        block_size,
                        grid_size);

  }

  for(size_t i=0; i<_Nx*_Ny*_Nz; i++) {
    _Ex_simu[i] = Ex_temp[i];
    _Ey_simu[i] = Ey_temp[i];
    _Ez_simu[i] = Ez_temp[i];
    _Hx_simu[i] = Hx_temp[i];
    _Hy_simu[i] = Hy_temp[i];
    _Hz_simu[i] = Hz_temp[i];
  }

} 


void gDiamond::_updateEH_phase_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
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
                         size_t grid_size) 
{
  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num);
  
    for(size_t t=0; t<BLT_GPU; t++) {
      
      int calculate_Ex = 1; // calculate this E tile or not
      int calculate_Hx = 1; // calculate this H tile or not
      int calculate_Ey = 1; 
      int calculate_Hy = 1; 
      int calculate_Ez = 1; 
      int calculate_Hz = 1; 
  
      // {Ehead, Etail, Hhead, Htail}
      std::vector<int> indices_X = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  xx_heads, xx_tails,
                                                  xx, t,
                                                  m_or_v_X,
                                                  Nx,
                                                  &calculate_Ex, &calculate_Hx);

      /*
      if(yy == 0 && zz == 0) {
        std::cout << "X dimension, xx = " << xx << ", t = " << t << "\n";
        std::cout << "calculate_Ex = " << calculate_Ex << ", calculate_Hx = " << calculate_Hx << ", ";
        std::cout << "Ehead = " << indices_X[0] << ", " 
                  << "Etail = " << indices_X[1] << ", "
                  << "Hhead = " << indices_X[2] << ", "
                  << "Htail = " << indices_X[3] << "\n";
                                      
      }
      */

      std::vector<int> indices_Y = _get_head_tail(BLY_GPU, BLT_GPU,
                                                  yy_heads, yy_tails,
                                                  yy, t,
                                                  m_or_v_Y,
                                                  Ny,
                                                  &calculate_Ey, &calculate_Hy);

      std::vector<int> indices_Z = _get_head_tail(BLZ_GPU, BLT_GPU,
                                                  zz_heads, zz_tails,
                                                  zz, t,
                                                  m_or_v_Z,
                                                  Nz,
                                                  &calculate_Ez, &calculate_Hz);

      /*

      // update E
      if(calculate_Ex & calculate_Ey & calculate_Ez) {
        for(int x=indices_X[0]; x<=indices_X[1]; x++) {
          for(int y=indices_Y[0]; y<=indices_Y[1]; y++) {
            for(int z=indices_Z[0]; z<=indices_Z[1]; z++) {
              if(x >= 1 && x <= Nx - 2 && y >= 1 && y <= Ny - 2 && z >= 1 && z <= Nz - 2) {
                int g_idx = x + y * Nx + z * Nx * Ny; // global idx

                Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                          ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
                Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                          ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
                Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                          ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
              }
            }
          }
        }
      }

      // update H
      if(calculate_Hx & calculate_Hy & calculate_Hz) {
        for(int x=indices_X[2]; x<=indices_X[3]; x++) {
          for(int y=indices_Y[2]; y<=indices_Y[3]; y++) {
            for(int z=indices_Z[2]; z<=indices_Z[3]; z++) {
              if(x >= 1 && x <= Nx - 2 && y >= 1 && y <= Ny - 2 && z >= 1 && z <= Nz - 2) {
                int g_idx = x + y * Nx + z * Nx * Ny; // global idx

                Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                          ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
                Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                          ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
                Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                          ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
              }
            }
          }
        }
      }
      */

      // update E
      if(calculate_Ex & calculate_Ey & calculate_Ez) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Ehead is offset
          int global_x = indices_X[0] + local_x; // Global X coordinate
          int global_y = indices_Y[0] + local_y; // Global Y coordinate
          int global_z = indices_Z[0] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[1] &&
            global_y <= indices_Y[1] &&
            global_z <= indices_Z[1]) {
            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                      ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
            Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                      ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
            Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                      ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
          }
        }
      }

      // update H 
      if(calculate_Hx & calculate_Hy & calculate_Hz) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Hhead is offset
          int global_x = indices_X[2] + local_x; // Global X coordinate
          int global_y = indices_Y[2] + local_y; // Global Y coordinate
          int global_z = indices_Z[2] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[3] &&
            global_y <= indices_Y[3] &&
            global_z <= indices_Z[3]) {
            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                      ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
            Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                      ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
            Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                      ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
          }
        }
      }

    }
  } 

}



void gDiamond::update_FDTD_gpu_3D_warp_underutilization(size_t num_timesteps) {

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  CUDACHECK(cudaMalloc(&Ex, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ey, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ez, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&My, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cay, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Caz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Day, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Daz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * _Nx * _Ny * _Nz)); 

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpy(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // set block and grid
  dim3 blockSize_3D(BLOCK_SIZE, 1, 1);
  dim3 gridSize_3D((_Nx + blockSize_3D.x - 1) / blockSize_3D.x, _Ny, _Nz);

  for(size_t t=0; t<num_timesteps; t++) {

    // Current source
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_3Dmap<<<gridSize_3D, blockSize_3D, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_3Dmap<<<gridSize_3D, blockSize_3D, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, _dx, _Nx, _Ny, _Nz);

  }
  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  CUDACHECK(cudaFree(Ex));
  CUDACHECK(cudaFree(Ey));
  CUDACHECK(cudaFree(Ez));
  CUDACHECK(cudaFree(Hx));
  CUDACHECK(cudaFree(Hy));
  CUDACHECK(cudaFree(Hz));
  CUDACHECK(cudaFree(Jx));
  CUDACHECK(cudaFree(Jy));
  CUDACHECK(cudaFree(Jz));
  CUDACHECK(cudaFree(Mx));
  CUDACHECK(cudaFree(My));
  CUDACHECK(cudaFree(Mz));
  CUDACHECK(cudaFree(Cax));
  CUDACHECK(cudaFree(Cbx));
  CUDACHECK(cudaFree(Cay));
  CUDACHECK(cudaFree(Cby));
  CUDACHECK(cudaFree(Caz));
  CUDACHECK(cudaFree(Cbz));
  CUDACHECK(cudaFree(Dax));
  CUDACHECK(cudaFree(Dbx));
  CUDACHECK(cudaFree(Day));
  CUDACHECK(cudaFree(Dby));
  CUDACHECK(cudaFree(Daz));
  CUDACHECK(cudaFree(Dbz));
}

void gDiamond::update_FDTD_gpu_fuse_kernel(size_t num_timesteps) { // 3-D mapping, using diamond tiling to fuse kernels

  // get the size of shared memory
  int device;
  cudaGetDevice(&device); // Get the currently active device
  int sharedMemoryPerBlock;
  int sharedMemoryPerSM;
  cudaDeviceGetAttribute(&sharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
  cudaDeviceGetAttribute(&sharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  std::cout << "maximum shared memory per block: " << sharedMemoryPerBlock << " bytes" << std::endl;
  std::cout << "maximum num of floats per block: " << sharedMemoryPerBlock / sizeof(float) << "\n";
  std::cout << "maximum shared memory per SM: " << sharedMemoryPerSM << " bytes" << std::endl;

  /*
    for Nx = Ny = Nz = 100
    we set BLX = BLY = BLZ, then BLX = BLY = BLZ = 8.
  */

  // we don't care about different ranges within BLT
  // cuz for GPU, if we don't calculate, threads will be idling anyways
  // find ranges for mountains in X dimension
  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[1][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[1][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *mountain_heads_Z_d, *mountain_tails_Z_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;
  int *valley_heads_Z_d, *valley_tails_Z_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_heads_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&mountain_tails_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_heads_Z_d, sizeof(int) * num_valleys_Z));
  CUDACHECK(cudaMalloc(&valley_tails_Z_d, sizeof(int) * num_valleys_Z));

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  CUDACHECK(cudaMalloc(&Ex, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ey, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ez, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&My, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cay, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Caz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Day, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Daz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * _Nx * _Ny * _Nz)); 

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Z_d, mountain_heads_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Z_d, mountain_tails_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Z_d, valley_heads_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Z_d, valley_tails_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
  // size_t shared_memory_size = (BLX_GPU * BLY_GPU * BLZ_GPU * 12 +
  //                             (BLX_GPU + 1) * (BLY_GPU + 1) * (BLZ_GPU + 1) * 6) * 4;  
  size_t shared_memory_size = BLX_EH * BLY_EH * BLZ_EH * 6 * sizeof(float);  
  std::cout << "grid_size = " << grid_size << "\n";
  std::cout << "block_size = " << block_size << "\n";
  std::cout << "shared_memory_size = " << shared_memory_size << "\n";

  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
    // grid size is changing for each phase

    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx, 
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d, 
                                   mountain_heads_Y_d, 
                                   mountain_heads_Z_d,
                                   mountain_tails_X_d, 
                                   mountain_tails_Y_d, 
                                   mountain_tails_Z_d
                                   );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "phase 1 kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // phase 2: (v, m, m) 
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx, 
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                   valley_heads_X_d, 
                                   mountain_heads_Y_d, 
                                   mountain_heads_Z_d,
                                   valley_tails_X_d, 
                                   mountain_tails_Y_d, 
                                   mountain_tails_Z_d
                                   );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "phase 2 kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx, 
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d, 
                                   valley_heads_Y_d, 
                                   mountain_heads_Z_d,
                                   mountain_tails_X_d, 
                                   valley_tails_Y_d, 
                                   mountain_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 3 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 4: (m, m, v)
  grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
  updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d,
                                   mountain_heads_Y_d,
                                   valley_heads_Z_d,
                                   mountain_tails_X_d,
                                   mountain_tails_Y_d,
                                   valley_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 4 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 5: (v, v, m)
  grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
  updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                   valley_heads_X_d,
                                   valley_heads_Y_d,
                                   mountain_heads_Z_d,
                                   valley_tails_X_d,
                                   valley_tails_Y_d,
                                   mountain_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 5 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 6: (v, m, v)
  grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
  updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                   valley_heads_X_d,
                                   mountain_heads_Y_d,
                                   valley_heads_Z_d,
                                   valley_tails_X_d,
                                   mountain_tails_Y_d,
                                   valley_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 6 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 7: (m, v, v)
  grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
  updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d,
                                   mountain_tails_X_d,
                                   valley_tails_Y_d,
                                   valley_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 7 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 8: (v, v, v)
  grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
  updateEH_phase_EH_shared_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   valley_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d,
                                   valley_tails_X_d,
                                   valley_tails_Y_d,
                                   valley_tails_Z_d
                                   );
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 8 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }
  }
  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

  CUDACHECK(cudaFree(Ex));
  CUDACHECK(cudaFree(Ey));
  CUDACHECK(cudaFree(Ez));
  CUDACHECK(cudaFree(Hx));
  CUDACHECK(cudaFree(Hy));
  CUDACHECK(cudaFree(Hz));
  CUDACHECK(cudaFree(Jx));
  CUDACHECK(cudaFree(Jy));
  CUDACHECK(cudaFree(Jz));
  CUDACHECK(cudaFree(Mx));
  CUDACHECK(cudaFree(My));
  CUDACHECK(cudaFree(Mz));
  CUDACHECK(cudaFree(Cax));
  CUDACHECK(cudaFree(Cbx));
  CUDACHECK(cudaFree(Cay));
  CUDACHECK(cudaFree(Cby));
  CUDACHECK(cudaFree(Caz));
  CUDACHECK(cudaFree(Cbz));
  CUDACHECK(cudaFree(Dax));
  CUDACHECK(cudaFree(Dbx));
  CUDACHECK(cudaFree(Day));
  CUDACHECK(cudaFree(Dby));
  CUDACHECK(cudaFree(Daz));
  CUDACHECK(cudaFree(Dbz));

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(mountain_heads_Z_d));
  CUDACHECK(cudaFree(mountain_tails_Z_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_Z_d));
  CUDACHECK(cudaFree(valley_tails_Z_d));

}

void gDiamond::update_FDTD_gpu_fuse_kernel_testing(size_t num_timesteps) { // 3-D mapping, using diamond tiling to fuse kernels

  // get the size of shared memory
  int device;
  cudaGetDevice(&device); // Get the currently active device
  int sharedMemoryPerBlock;
  int sharedMemoryPerSM;
  cudaDeviceGetAttribute(&sharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
  cudaDeviceGetAttribute(&sharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  std::cout << "maximum shared memory per block: " << sharedMemoryPerBlock << " bytes" << std::endl;
  std::cout << "maximum num of floats per block: " << sharedMemoryPerBlock / sizeof(float) << "\n";
  std::cout << "maximum shared memory per SM: " << sharedMemoryPerSM << " bytes" << std::endl;

  /*
    for Nx = Ny = Nz = 100
    we set BLX = BLY = BLZ, then BLX = BLY = BLZ = 8.
  */

  // we don't care about different ranges within BLT
  // cuz for GPU, if we don't calculate, threads will be idling anyways
  // find ranges for mountains in X dimension
  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[1][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[1][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *mountain_heads_Z_d, *mountain_tails_Z_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;
  int *valley_heads_Z_d, *valley_tails_Z_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_heads_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&mountain_tails_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_heads_Z_d, sizeof(int) * num_valleys_Z));
  CUDACHECK(cudaMalloc(&valley_tails_Z_d, sizeof(int) * num_valleys_Z));

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  CUDACHECK(cudaMalloc(&Ex, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ey, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ez, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&My, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cay, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Caz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Day, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Daz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * _Nx * _Ny * _Nz)); 

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Z_d, mountain_heads_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Z_d, mountain_tails_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Z_d, valley_heads_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Z_d, valley_tails_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
  // size_t shared_memory_size = (BLX_GPU * BLY_GPU * BLZ_GPU * 12 +
  //                             (BLX_GPU + 1) * (BLY_GPU + 1) * (BLZ_GPU + 1) * 6) * 4;  
  size_t shared_memory_size = BLX_EH * BLY_EH * BLZ_EH * 6 * sizeof(float);  
  std::cout << "grid_size = " << grid_size << "\n";
  std::cout << "block_size = " << block_size << "\n";
  std::cout << "shared_memory_size = " << shared_memory_size << "\n";

  for(size_t tt=0; tt<num_timesteps/BLT_GPU; tt++) {
    // grid size is changing for each phase

    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    for(size_t t=0; t<BLT_GPU; t++) {
      updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d, 
                                     mountain_heads_Y_d, 
                                     mountain_heads_Z_d);
      updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d, 
                                     mountain_heads_Y_d, 
                                     mountain_heads_Z_d);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "phase 1 kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // phase 2: (v, m, m) 
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    for(size_t t=0; t<BLT_GPU; t++) {
      updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                     valley_heads_X_d, 
                                     mountain_heads_Y_d, 
                                     mountain_heads_Z_d);
      updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_mountains_Y, num_mountains_Z, // number of tiles in each dimensions
                                     valley_heads_X_d, 
                                     mountain_heads_Y_d, 
                                     mountain_heads_Z_d);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "phase 2 kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    for(size_t t=0; t<BLT_GPU; t++) {
      updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d, 
                                     valley_heads_Y_d, 
                                     mountain_heads_Z_d);
      updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx, 
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d, 
                                     valley_heads_Y_d, 
                                     mountain_heads_Z_d);
   }
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 3 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 4: (m, m, v)
  grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
  for(size_t t=0; t<BLT_GPU; t++) {
    updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d,
                                     mountain_heads_Y_d,
                                     valley_heads_Z_d);
    updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_mountains_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                     mountain_heads_X_d,
                                     mountain_heads_Y_d,
                                     valley_heads_Z_d);
   } 
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 4 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 5: (v, v, m)
  grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
  for(size_t t=0; t<BLT_GPU; t++) {
    updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                     valley_heads_X_d,
                                     valley_heads_Y_d,
                                     mountain_heads_Z_d);
    updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_valleys_Y, num_mountains_Z, // number of tiles in each dimensions
                                     valley_heads_X_d,
                                     valley_heads_Y_d,
                                     mountain_heads_Z_d);
  } 
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "phase 5 kernel launch error: " << cudaGetErrorString(err) << std::endl;
  }

  // phase 6: (v, m, v)
  grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
  for(size_t t=0; t<BLT_GPU; t++) {
    updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                     valley_heads_X_d,
                                     mountain_heads_Y_d,
                                     valley_heads_Z_d);
    updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                     Hx, Hy, Hz,
                                     Cax, Cbx,
                                     Cay, Cby,
                                     Caz, Cbz,
                                     Dax, Dbx,
                                     Day, Dby,
                                     Daz, Dbz,
                                     Jx, Jy, Jz,
                                     Mx, My, Mz,
                                     _dx,
                                     _Nx, _Ny, _Nz,
                                     num_valleys_X, num_mountains_Y, num_valleys_Z, // number of tiles in each dimensions
                                     valley_heads_X_d,
                                     mountain_heads_Y_d,
                                     valley_heads_Z_d);
   } 
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 6 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 7: (m, v, v)
  grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
  for(size_t t=0; t<BLT_GPU; t++) {
  updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d);
  updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_mountains_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   mountain_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d);
   } 
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 7 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }

  // phase 8: (v, v, v)
  grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
  for(size_t t=0; t<BLT_GPU; t++) {
  updateEH_phase_E_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   valley_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d);
  updateEH_phase_H_only<<<grid_size, block_size, shared_memory_size>>>(Ex, Ey, Ez,
                                   Hx, Hy, Hz,
                                   Cax, Cbx,
                                   Cay, Cby,
                                   Caz, Cbz,
                                   Dax, Dbx,
                                   Day, Dby,
                                   Daz, Dbz,
                                   Jx, Jy, Jz,
                                   Mx, My, Mz,
                                   _dx,
                                   _Nx, _Ny, _Nz,
                                   num_valleys_X, num_valleys_Y, num_valleys_Z, // number of tiles in each dimensions
                                   valley_heads_X_d,
                                   valley_heads_Y_d,
                                   valley_heads_Z_d);
   } 
   err = cudaGetLastError();
   if (err != cudaSuccess) {
     std::cerr << "phase 8 kernel launch error: " << cudaGetErrorString(err) << std::endl;
   }
  }
  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

  CUDACHECK(cudaFree(Ex));
  CUDACHECK(cudaFree(Ey));
  CUDACHECK(cudaFree(Ez));
  CUDACHECK(cudaFree(Hx));
  CUDACHECK(cudaFree(Hy));
  CUDACHECK(cudaFree(Hz));
  CUDACHECK(cudaFree(Jx));
  CUDACHECK(cudaFree(Jy));
  CUDACHECK(cudaFree(Jz));
  CUDACHECK(cudaFree(Mx));
  CUDACHECK(cudaFree(My));
  CUDACHECK(cudaFree(Mz));
  CUDACHECK(cudaFree(Cax));
  CUDACHECK(cudaFree(Cbx));
  CUDACHECK(cudaFree(Cay));
  CUDACHECK(cudaFree(Cby));
  CUDACHECK(cudaFree(Caz));
  CUDACHECK(cudaFree(Cbz));
  CUDACHECK(cudaFree(Dax));
  CUDACHECK(cudaFree(Dbx));
  CUDACHECK(cudaFree(Day));
  CUDACHECK(cudaFree(Dby));
  CUDACHECK(cudaFree(Daz));
  CUDACHECK(cudaFree(Dbz));

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(mountain_heads_Z_d));
  CUDACHECK(cudaFree(mountain_tails_Z_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_Z_d));
  CUDACHECK(cudaFree(valley_tails_Z_d));



}

void gDiamond::_updateEH_phase_E_only_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
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
                               size_t grid_size)
{
  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num);
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
      int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
      int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

      int global_x = xx_heads[xx] + local_x; // Global X coordinate
      int global_y = yy_heads[yy] + local_y; // Global Y coordinate
      int global_z = zz_heads[zz] + local_z; // Global Z coordinate

      if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
         local_x >= xx_heads[xx] && local_x <= xx_tails[xx] &&
         local_y >= yy_heads[yy] && local_y <= yy_tails[yy] &&
         local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {
        int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

        // update E
        Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                  ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
        Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                  ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
        Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                  ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
      }
    }
  }
}

void gDiamond::_updateEH_phase_H_only_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
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
                               size_t grid_size)
                       {
  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num);
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
      int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
      int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

      int global_x = xx_heads[xx] + local_x; // Global X coordinate
      int global_y = yy_heads[yy] + local_y; // Global Y coordinate
      int global_z = zz_heads[zz] + local_z; // Global Z coordinate

      if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
         local_x >= xx_heads[xx] && local_x <= xx_tails[xx] &&
         local_y >= yy_heads[yy] && local_y <= yy_tails[yy] &&
         local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {
        int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

        // update H
        Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                  ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
        Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                  ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
        Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                  ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
      }
    }
  }
}

