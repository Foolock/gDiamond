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
./example/seq_figures Nx Ny Nz num_timesteps # CPU sequential example
./example/omp_figures Nx Ny Nz num_timesteps # CPU parallel example with openmp
./example/gpu_figures Nx Ny Nz num_timesteps # GPU example 
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


