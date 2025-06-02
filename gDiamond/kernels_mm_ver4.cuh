// mix mapping version 4, one-to-one mapping on X and Y and Z
// break big mountain tile into one small mountain and multiple parallelograms
#ifndef KERNELS_MM_VER4_CUH
#define KERNELS_MM_VER4_CUH

#include "gdiamond.hpp"

#define BLT_MM_V4 4 

/*
 * X dimension parameters
 */

// number of threads used in X dimension
#define NTX_MM_V4 16
// width of mountain bottom of the small mountain (replication part) in X 
#define BLX_R 8
// width of parallelogram in X 
#define BLX_P NTX_MM_V4
// number of parallelgrams in each big mountain in X
#define NUM_P_X 1
// width of mountain bottom of the big mountain in X
#define MOUNTAIN_X_V4 (BLX_R + NUM_P_X * BLX_P)
#define VALLEY_X_V4 (MOUNTAIN_X_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

/*
 * Y dimension parameters
 */

// number of threads used in Y dimension
#define NTY_MM_V4 8 
// width of mountain bottom of the small mountain (replication part) in Y 
#define BLY_R 8
// width of parallelogram in Y 
#define BLY_P NTY_MM_V4
// number of parallelgrams in each big mountain in Y
#define NUM_P_Y 2 
// width of mountain bottom of the big mountain in Y
#define MOUNTAIN_Y_V4 (BLY_R + NUM_P_Y * BLY_P)
#define VALLEY_Y_V4 (MOUNTAIN_Y_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

/*
 * Z dimension parameters
 */

// number of threads used in Z dimension
#define NTZ_MM_V4 8 
// width of mountain bottom of the small mountain (replication part) in Z 
#define BLZ_R 8
// width of parallelogram in Z 
#define BLZ_P NTZ_MM_V4
// number of parallelgrams in each big mountain in Z
#define NUM_P_Z 2 
// width of mountain bottom of the big mountain in Z
#define MOUNTAIN_Z_V4 (BLZ_R + NUM_P_Z * BLZ_P)
#define VALLEY_Z_V4 (MOUNTAIN_Z_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

// padding
#define LEFT_PAD_MM_V4 BLT_MM_V4
#define RIGHT_PAD_MM_V4 BLT_MM_V4

// shared memory size
#define H_SHX_V4 (BLX_P + BLT_MM_V4) 
#define H_SHY_V4 (BLY_P + BLT_MM_V4) 
#define H_SHZ_V4 (BLZ_P + BLT_MM_V4) 
#define E_SHX_V4 (BLX_P + BLT_MM_V4)
#define E_SHY_V4 (BLY_P + BLT_MM_V4)
#define E_SHZ_V4 (BLZ_P + BLT_MM_V4)



#endif


































