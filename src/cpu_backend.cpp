#include "reaction_diffusion.h"

#include <algorithm>
#include <chrono>
#include <ostream>
#include <vector>

namespace rd {
namespace {

float ClampFloat(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

int Wrap(int value, int limit) {
  if (value < 0) {
    return value + limit;
  }
  if (value >= limit) {
    return value - limit;
  }
  return value;
}

const Cell& At(const std::vector<Cell>& cells, int width, int height, int x,
               int y) {
  const int wrapped_x = Wrap(x, width);
  const int wrapped_y = Wrap(y, height);
  return cells[static_cast<size_t>(wrapped_y) * width + wrapped_x];
}

float LaplacianA(const std::vector<Cell>& cells, int width, int height, int x,
                 int y) {
  const float center = At(cells, width, height, x, y).a;
  const float cardinal = At(cells, width, height, x - 1, y).a +
                         At(cells, width, height, x + 1, y).a +
                         At(cells, width, height, x, y - 1).a +
                         At(cells, width, height, x, y + 1).a;
  const float diagonal = At(cells, width, height, x - 1, y - 1).a +
                         At(cells, width, height, x + 1, y - 1).a +
                         At(cells, width, height, x - 1, y + 1).a +
                         At(cells, width, height, x + 1, y + 1).a;
  return -center + 0.2f * cardinal + 0.05f * diagonal;
}

float LaplacianB(const std::vector<Cell>& cells, int width, int height, int x,
                 int y) {
  const float center = At(cells, width, height, x, y).b;
  const float cardinal = At(cells, width, height, x - 1, y).b +
                         At(cells, width, height, x + 1, y).b +
                         At(cells, width, height, x, y - 1).b +
                         At(cells, width, height, x, y + 1).b;
  const float diagonal = At(cells, width, height, x - 1, y - 1).b +
                         At(cells, width, height, x + 1, y - 1).b +
                         At(cells, width, height, x - 1, y + 1).b +
                         At(cells, width, height, x + 1, y + 1).b;
  return -center + 0.2f * cardinal + 0.05f * diagonal;
}

void Step(const Config& config, const std::vector<Cell>& current,
          std::vector<Cell>* next) {
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      const size_t index = static_cast<size_t>(y) * config.width + x;
      const Cell cell = current[index];
      const float reaction = cell.a * cell.b * cell.b;
      const float next_a =
          cell.a + (config.diffusion_a *
                        LaplacianA(current, config.width, config.height, x, y) -
                    reaction + config.feed * (1.0f - cell.a)) *
                       config.dt;
      const float next_b =
          cell.b + (config.diffusion_b *
                        LaplacianB(current, config.width, config.height, x, y) +
                    reaction - (config.kill + config.feed) * cell.b) *
                       config.dt;
      (*next)[index].a = ClampFloat(next_a, 0.0f, 1.0f);
      (*next)[index].b = ClampFloat(next_b, 0.0f, 1.0f);
    }
  }
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

bool EmitProof(const Config& config, const std::vector<Cell>& cells, int step,
               double elapsed_ms, std::string* error) {
  if (!WriteFrame(cells, config, step, error)) {
    return false;
  }
  return AppendMetricsRow(config, ComputeMetrics(cells, step, elapsed_ms),
                          error);
}

}  // namespace

bool RunCpuSimulation(const Config& config, std::ostream* log,
                      std::string* error) {
  if (!EnsureOutputDirectory(config.output_dir, error)) {
    return false;
  }
  if (!WriteMetricsHeader(config, error)) {
    return false;
  }

  std::vector<Cell> current = BuildInitialGrid(config);
  std::vector<Cell> next(current.size());

  if (log != nullptr) {
    *log << "backend=cpu\n"
         << "grid=" << config.width << "x" << config.height << '\n'
         << "steps=" << config.steps << '\n'
         << "preset=" << config.preset << '\n'
         << "output_dir=" << config.output_dir << '\n';
  }

  const auto start = std::chrono::steady_clock::now();
  if (!EmitProof(config, current, 0, 0.0, error)) {
    return false;
  }

  for (int step = 1; step <= config.steps; ++step) {
    Step(config, current, &next);
    current.swap(next);

    if (step % config.output_interval == 0 || step == config.steps) {
      const double elapsed_ms = ElapsedMs(start);
      if (!EmitProof(config, current, step, elapsed_ms, error)) {
        return false;
      }
      if (log != nullptr) {
        *log << "wrote proof for step=" << step
             << " elapsed_ms=" << elapsed_ms << '\n';
      }
    }
  }

  if (log != nullptr) {
    *log << "status=ok\n";
  }
  return true;
}

}  // namespace rd
