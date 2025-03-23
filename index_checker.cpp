#include <iostream>
#include <vector>
#include <tuple>

#define BLX_UB 1 
#define BLY_UB 32 
#define BLZ_UB 16

const int Nx = 32;  // Number of elements along X-axis
const int Ny = 32;  // Number of elements along Y-axis
const int Nz = 32;  // Number of elements along Z-axis

std::tuple<int, int, int> naive_mapping(int thread_id, int Nx, int Ny, int Nz) {
    int x = thread_id % Nx;
    int y = (thread_id % (Nx * Ny)) / Nx;
    int z = thread_id / (Nx * Ny);
    return {x, y, z};
}

std::tuple<int, int, int> tile_mapping1(int xx_num, int yy_num, int zz_num,
                                       std::vector<int> xx_heads, std::vector<int> yy_heads, std::vector<int> zz_heads,
                                       int block_id, int thread_id, int Nx, int Ny, int Nz) {

  // Compute the tile (xx, yy, zz) that this block is responsible for
  int xx = block_id % xx_num;
  int yy = (block_id / xx_num) % yy_num;
  int zz = block_id / (xx_num * yy_num);

  // Compute local thread coordinates within the tile
  int local_x = thread_id % BLX_UB;
  int local_y = (thread_id / BLX_UB) % BLY_UB;
  int local_z = thread_id / (BLX_UB * BLY_UB);

  // Compute global indices based on tile offsets
  int global_x = xx_heads[xx] + local_x;
  int global_y = yy_heads[yy] + local_y;
  int global_z = zz_heads[zz] + local_z;

  std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << ", ";
  std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << ", ";
  std::cout << "xx = " << xx << ", yy = " << yy << ", zz = " << zz << "\n";

  return {global_x, global_y, global_z};
}

std::tuple<int, int, int> tile_mapping2(int xx_num, int yy_num, int zz_num,
                                       std::vector<int> xx_heads, std::vector<int> yy_heads, std::vector<int> zz_heads,
                                       int block_id, int thread_id, int Nx, int Ny, int Nz) {

  // Compute x, y, z in the same order as naive mapping
  int x = thread_id % Nx;
  int y = (thread_id % (Nx * Ny)) / Nx;
  int z = thread_id / (Nx * Ny);

  // Find which tile (xx, yy, zz) the thread belongs to
  int xx = x / BLX_UB;
  int yy = y / BLY_UB;
  int zz = z / BLZ_UB;

  std::cout << "xx = " << xx << ", yy = " << yy << ", zz = " << zz << "\n";

  // Compute local position inside the tile
  int local_x = x % BLX_UB;
  int local_y = y % BLY_UB;
  int local_z = z % BLZ_UB;

  // Compute global indices based on tile offsets
  int global_x = xx_heads[xx] + local_x;
  int global_y = yy_heads[yy] + local_y;
  int global_z = zz_heads[zz] + local_z;
 
  return {global_x, global_y, global_z};
}

int main() {

    // get xx_num, yy_num, zz_num
    int xx_num = (Nx + BLX_UB - 1) / BLX_UB;
    int yy_num = (Ny + BLY_UB - 1) / BLY_UB;
    int zz_num = (Nz + BLZ_UB - 1) / BLZ_UB;

    // get xx_heads, yy_heads, zz_heads
    std::vector<int> xx_heads(xx_num, 0);
    std::vector<int> yy_heads(yy_num, 0);
    std::vector<int> zz_heads(zz_num, 0);
    for(int i=0; i<xx_num; i++) {
      xx_heads[i] = i * BLX_UB;
    }
    for(int i=0; i<yy_num; i++) {
      yy_heads[i] = i * BLY_UB;
    }
    for(int i=0; i<zz_num; i++) {
      zz_heads[i] = i * BLZ_UB;
    }

    int block_size = 512;
    int grid_size;

    std::cout << "naive mapping\n";
    grid_size = (Nx * Ny * Nz + block_size - 1) / block_size;
    for (int block_id = 0; block_id < grid_size; ++block_id) {
      std::cout << "block_id = " << block_id << "\n";
      for (int thread_id = 0; thread_id < 512; ++thread_id) {
          auto [x, y, z] = naive_mapping(block_id * block_size + thread_id, Nx, Ny, Nz);
          int global_id = x + y * Nx + z * (Nx * Ny); 
          std::cout << "1D Index: " << thread_id << " -> global_id: " << global_id << ", global_x = " << x << ", global_y = " << y << ", global_z = " << z << "\n";
      }
      std::cout << "--------------------------------------------------------\n\n";
    }

    std::cout << "tile mapping1\n";
    grid_size = xx_num * yy_num * zz_num;
    for (int block_id = 0; block_id < grid_size; ++block_id) {
      std::cout << "block_id = " << block_id << "\n";
      for (int thread_id = 0; thread_id < 512; ++thread_id) {
          auto [x, y, z] = tile_mapping1(xx_num, yy_num, zz_num,
                                        xx_heads, yy_heads, zz_heads,
                                        block_id, thread_id, Nx, Ny, Nz);
          int global_id = x + y * Nx + z * (Nx * Ny); 
          std::cout << "1D Index: " << thread_id << " -> global_id: " << global_id << "\n";
      }
      std::cout << "--------------------------------------------------------\n\n";
    }

    // std::cout << "tile mapping2\n";
    // grid_size = xx_num * yy_num * zz_num;
    // for (int block_id = 0; block_id < grid_size; ++block_id) {
    //   std::cout << "block_id = " << block_id << "\n";
    //   for (int thread_id = 0; thread_id < 512; ++thread_id) {
    //       auto [x, y, z] = tile_mapping2(xx_num, yy_num, zz_num,
    //                                     xx_heads, yy_heads, zz_heads,
    //                                     block_id, thread_id, Nx, Ny, Nz);
    //       int global_id = x + y * Nx + z * (Nx * Ny); 
    //       std::cout << "1D Index: " << thread_id << " -> global_id: " << global_id << "\n";
    //   }
    //   std::cout << "--------------------------------------------------------\n\n";
    // }

    return 0;
}
































