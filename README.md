# gDiamond

### To build the project
```bash
git clone https://github.com/Foolock/gDiamond.git
cd gDiamond
mkdir build
cd build
cmake ../
make -j 16 # plz ignore warnings since it is under development
```

### To run examples with figures output
```bash
cd build # after you build the project
./examples/seq_figures Nx Ny Nz num_timesteps # CPU sequential example
./examples/omp_figures Nx Ny Nz num_timesteps # CPU parallel example with openmp
./examples/gpu_figures Nx Ny Nz num_timesteps # GPU example 
```

### Environment info
```bash
cmake --version
# cmake version 3.22.1
g++ --version
# g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
nvidia-smi
# NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2
```

### Note 
1. (5/8) BLT = 3 produces better results than BLT = 4 in mix mapping ver2. Might switch block\_size from 1024 to 768.
