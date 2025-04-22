#ifndef KERNELS_MM_CUH
#define KERNELS_MM_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM 4

// one-to-one mapping in X dimension
#define NTX_MM 16
#define MOUNTAIN_X 16 
#define VALLEY_X (MOUNTAIN_X - 2 * (BLT_MM - 1) - 1) 

// one-to-many mapping in Y dimension
#define NTY_MM 4
#define MOUNTAIN_Y 10
#define VALLEY_Y (MOUNTAIN_Y - 2 * (BLT_MM - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM 4
#define MOUNTAIN_Z 10
#define VALLEY_Z (MOUNTAIN_Z - 2 * (BLT_MM - 1) - 1)

// padding
#define LEFT_PAD_MM BLT_MM
#define RIGHT_PAD_MM BLT_MM

#endif
