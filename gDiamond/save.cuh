// store copy of implementations

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

