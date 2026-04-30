#include "reaction_diffusion.h"

#include <cuda_runtime.h>

#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rd {
namespace {

__device__ int WrapDevice(int value, int limit) {
  if (value < 0) {
    return value + limit;
  }
  if (value >= limit) {
    return value - limit;
  }
  return value;
}

__device__ Cell AtDevice(const Cell* cells, int width, int height, int x,
                         int y) {
  const int wrapped_x = WrapDevice(x, width);
  const int wrapped_y = WrapDevice(y, height);
  return cells[wrapped_y * width + wrapped_x];
}

__device__ float LaplacianADevice(const Cell* cells, int width, int height,
                                  int x, int y) {
  const float center = AtDevice(cells, width, height, x, y).a;
  const float cardinal = AtDevice(cells, width, height, x - 1, y).a +
                         AtDevice(cells, width, height, x + 1, y).a +
                         AtDevice(cells, width, height, x, y - 1).a +
                         AtDevice(cells, width, height, x, y + 1).a;
  const float diagonal = AtDevice(cells, width, height, x - 1, y - 1).a +
                         AtDevice(cells, width, height, x + 1, y - 1).a +
                         AtDevice(cells, width, height, x - 1, y + 1).a +
                         AtDevice(cells, width, height, x + 1, y + 1).a;
  return -center + 0.2f * cardinal + 0.05f * diagonal;
}

__device__ float LaplacianBDevice(const Cell* cells, int width, int height,
                                  int x, int y) {
  const float center = AtDevice(cells, width, height, x, y).b;
  const float cardinal = AtDevice(cells, width, height, x - 1, y).b +
                         AtDevice(cells, width, height, x + 1, y).b +
                         AtDevice(cells, width, height, x, y - 1).b +
                         AtDevice(cells, width, height, x, y + 1).b;
  const float diagonal = AtDevice(cells, width, height, x - 1, y - 1).b +
                         AtDevice(cells, width, height, x + 1, y - 1).b +
                         AtDevice(cells, width, height, x - 1, y + 1).b +
                         AtDevice(cells, width, height, x + 1, y + 1).b;
  return -center + 0.2f * cardinal + 0.05f * diagonal;
}

__global__ void UpdateKernel(const Cell* current, Cell* next, int width,
                             int height, float diffusion_a, float diffusion_b,
                             float feed, float kill, float dt) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int index = y * width + x;
  const Cell cell = current[index];
  const float reaction = cell.a * cell.b * cell.b;
  const float next_a =
      cell.a +
      (diffusion_a * LaplacianADevice(current, width, height, x, y) -
       reaction + feed * (1.0f - cell.a)) *
          dt;
  const float next_b =
      cell.b +
      (diffusion_b * LaplacianBDevice(current, width, height, x, y) +
       reaction - (kill + feed) * cell.b) *
          dt;
  next[index].a = fminf(fmaxf(next_a, 0.0f), 1.0f);
  next[index].b = fminf(fmaxf(next_b, 0.0f), 1.0f);
}

bool CheckCuda(cudaError_t status, const char* expression, std::string* error) {
  if (status == cudaSuccess) {
    return true;
  }
  std::ostringstream message;
  message << expression << " failed: " << cudaGetErrorString(status);
  *error = message.str();
  return false;
}

double ElapsedMs(cudaEvent_t start, cudaEvent_t stop, std::string* error) {
  if (!CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)", error)) {
    return -1.0;
  }
  if (!CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)",
                 error)) {
    return -1.0;
  }
  float elapsed_ms = 0.0f;
  if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                 "cudaEventElapsedTime", error)) {
    return -1.0;
  }
  return elapsed_ms;
}

bool EmitProof(const Config& config, const Cell* device_cells,
               std::vector<Cell>* host_cells, int step, double elapsed_ms,
               std::string* error) {
  const size_t bytes = host_cells->size() * sizeof(Cell);
  if (!CheckCuda(cudaMemcpy(host_cells->data(), device_cells, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(device to host)", error)) {
    return false;
  }
  if (!WriteFrame(*host_cells, config, step, error)) {
    return false;
  }
  return AppendMetricsRow(config, ComputeMetrics(*host_cells, step, elapsed_ms),
                          error);
}

}  // namespace

bool RunGpuSimulation(const Config& config, std::ostream* log,
                      std::string* error) {
  if (!EnsureOutputDirectory(config.output_dir, error)) {
    return false;
  }
  if (!WriteMetricsHeader(config, error)) {
    return false;
  }

  int device = 0;
  if (!CheckCuda(cudaGetDevice(&device), "cudaGetDevice", error)) {
    return false;
  }
  cudaDeviceProp properties;
  if (!CheckCuda(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties", error)) {
    return false;
  }

  std::vector<Cell> host_cells = BuildInitialGrid(config);
  const size_t cell_count = host_cells.size();
  const size_t bytes = cell_count * sizeof(Cell);

  Cell* device_current = nullptr;
  Cell* device_next = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  if (!CheckCuda(cudaMalloc(&device_current, bytes), "cudaMalloc(current)",
                 error) ||
      !CheckCuda(cudaMalloc(&device_next, bytes), "cudaMalloc(next)", error) ||
      !CheckCuda(cudaMemcpy(device_current, host_cells.data(), bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(host to device)", error) ||
      !CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)", error) ||
      !CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)", error) ||
      !CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)", error)) {
    cudaFree(device_current);
    cudaFree(device_next);
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    return false;
  }

  if (log != nullptr) {
    *log << "backend=gpu\n"
         << "device=" << properties.name << '\n'
         << "compute_capability=" << properties.major << '.'
         << properties.minor << '\n'
         << "grid=" << config.width << "x" << config.height << '\n'
         << "steps=" << config.steps << '\n'
         << "preset=" << config.preset << '\n'
         << "block_size=" << config.block_size << '\n'
         << "output_dir=" << config.output_dir << '\n';
  }

  if (!EmitProof(config, device_current, &host_cells, 0, 0.0, error)) {
    cudaFree(device_current);
    cudaFree(device_next);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return false;
  }

  const dim3 block(config.block_size, config.block_size);
  const dim3 grid((config.width + block.x - 1) / block.x,
                  (config.height + block.y - 1) / block.y);

  for (int step = 1; step <= config.steps; ++step) {
    UpdateKernel<<<grid, block>>>(device_current, device_next, config.width,
                                  config.height, config.diffusion_a,
                                  config.diffusion_b, config.feed, config.kill,
                                  config.dt);
    if (!CheckCuda(cudaGetLastError(), "UpdateKernel launch", error)) {
      cudaFree(device_current);
      cudaFree(device_next);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return false;
    }
    std::swap(device_current, device_next);

    if (step % config.output_interval == 0 || step == config.steps) {
      if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", error)) {
        cudaFree(device_current);
        cudaFree(device_next);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
      }
      const double elapsed_ms = ElapsedMs(start, stop, error);
      if (elapsed_ms < 0.0 ||
          !EmitProof(config, device_current, &host_cells, step, elapsed_ms,
                     error)) {
        cudaFree(device_current);
        cudaFree(device_next);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
      }
      if (log != nullptr) {
        *log << "wrote proof for step=" << step
             << " elapsed_ms=" << elapsed_ms << '\n';
      }
    }
  }

  cudaFree(device_current);
  cudaFree(device_next);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (log != nullptr) {
    *log << "status=ok\n";
  }
  return true;
}

}  // namespace rd
