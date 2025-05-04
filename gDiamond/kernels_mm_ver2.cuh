// mix mapping version 2, one-to-one mapping on X and Y and Z
#ifndef KERNELS_MM_VER2_CUH
#define KERNELS_MM_VER2_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM_V2 4

// one-to-one mapping in X dimension
#define NTX_MM_V2 16
#define MOUNTAIN_X_V2 16 
#define VALLEY_X_V2 (MOUNTAIN_X_V2 - 2 * (BLT_MM_V2 - 1) - 1) 

// one-to-one mapping in Y dimension
#define NTY_MM_V2 8 
#define MOUNTAIN_Y_V2 8 
#define VALLEY_Y_V2 (MOUNTAIN_Y_V2 - 2 * (BLT_MM_V2 - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM_V2 8 
#define MOUNTAIN_Z_V2 8 
#define VALLEY_Z_V2 (MOUNTAIN_Z_V2 - 2 * (BLT_MM_V2 - 1) - 1)

// padding
#define LEFT_PAD_MM_V2 BLT_MM_V2
#define RIGHT_PAD_MM_V2 BLT_MM_V2

// tile size
#define BLX_MM_V2 MOUNTAIN_X_V2 
#define BLY_MM_V2 MOUNTAIN_Y_V2 
#define BLZ_MM_V2 MOUNTAIN_Z_V2 

#endif


































