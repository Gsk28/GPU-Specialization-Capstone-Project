#ifndef SRC_REACTION_DIFFUSION_H_
#define SRC_REACTION_DIFFUSION_H_

#include <iosfwd>
#include <string>
#include <vector>

namespace rd {

struct Cell {
  float a;
  float b;
};

struct Config {
  int width = 1024;
  int height = 1024;
  int steps = 5000;
  int output_interval = 500;
  int block_size = 16;
  unsigned int seed = 7;
  float diffusion_a = 1.0f;
  float diffusion_b = 0.5f;
  float feed = 0.0545f;
  float kill = 0.062f;
  float dt = 1.0f;
  bool write_images = true;
  std::string backend = "gpu";
  std::string output_dir = "proof/gpu_demo";
  std::string preset = "coral";
};

struct Metrics {
  int step = 0;
  double mean_a = 0.0;
  double mean_b = 0.0;
  float min_b = 0.0f;
  float max_b = 0.0f;
  double elapsed_ms = 0.0;
};

std::string Usage(const char* program);
bool ParseArgs(int argc, char** argv, Config* config, std::string* error);
bool ApplyPreset(const std::string& preset, Config* config, std::string* error);

std::vector<Cell> BuildInitialGrid(const Config& config);
Metrics ComputeMetrics(const std::vector<Cell>& cells, int step,
                       double elapsed_ms);
bool EnsureOutputDirectory(const std::string& output_dir, std::string* error);
bool WriteFrame(const std::vector<Cell>& cells, const Config& config, int step,
                std::string* error);
bool WriteMetricsHeader(const Config& config, std::string* error);
bool AppendMetricsRow(const Config& config, const Metrics& metrics,
                      std::string* error);

bool RunCpuSimulation(const Config& config, std::ostream* log,
                      std::string* error);

#ifdef RD_ENABLE_CUDA
bool RunGpuSimulation(const Config& config, std::ostream* log,
                      std::string* error);
#endif

}  // namespace rd

#endif  // SRC_REACTION_DIFFUSION_H_
