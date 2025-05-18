// mix mapping version 2, one-to-one mapping on X and Y and Z
#ifndef KERNELS_MM_VER3_CUH
#define KERNELS_MM_VER3_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM_V3 3 

// one-to-one mapping in X dimension
#define NTX_MM_V3 16
#define MOUNTAIN_X_V3 16 
// valley is actually mountain top 
#define VALLEY_X_V3 (MOUNTAIN_X_V3 - 2 * (BLT_MM_V3 - 1) - 1) 

// one-to-one mapping in Y dimension
#define NTY_MM_V3 8 
#define MOUNTAIN_Y_V3 8 
#define VALLEY_Y_V3 (MOUNTAIN_Y_V3 - 2 * (BLT_MM_V3 - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM_V3 8 
#define MOUNTAIN_Z_V3 8 
#define VALLEY_Z_V3 (MOUNTAIN_Z_V3 - 2 * (BLT_MM_V3 - 1) - 1)

// padding
#define LEFT_PAD_MM_V3 BLT_MM_V3
#define RIGHT_PAD_MM_V3 BLT_MM_V3

// tile size
#define BLX_MM_V3 MOUNTAIN_X_V3 
#define BLY_MM_V3 MOUNTAIN_Y_V3 
#define BLZ_MM_V3 MOUNTAIN_Z_V3 

// shared memory size
#define H_SHX (BLX_MM_V3 + 1)
#define H_SHY (BLY_MM_V3 + 1)
#define H_SHZ (BLZ_MM_V3 + 1)
#define E_SHX BLX_MM_V3
#define E_SHY BLY_MM_V3
#define E_SHZ BLZ_MM_V3


#endif


































