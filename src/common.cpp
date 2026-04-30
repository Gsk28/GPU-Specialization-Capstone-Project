#include "reaction_diffusion.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace rd {
namespace {

constexpr float kPi = 3.14159265358979323846f;

bool ParseInt(const std::string& value, const std::string& name, int* output,
              std::string* error) {
  try {
    size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size()) {
      *error = "Invalid integer for " + name + ": " + value;
      return false;
    }
    *output = parsed;
    return true;
  } catch (const std::exception&) {
    *error = "Invalid integer for " + name + ": " + value;
    return false;
  }
}

bool ParseUnsigned(const std::string& value, const std::string& name,
                   unsigned int* output, std::string* error) {
  try {
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size()) {
      *error = "Invalid unsigned integer for " + name + ": " + value;
      return false;
    }
    *output = static_cast<unsigned int>(parsed);
    return true;
  } catch (const std::exception&) {
    *error = "Invalid unsigned integer for " + name + ": " + value;
    return false;
  }
}

bool ParseFloat(const std::string& value, const std::string& name, float* output,
                std::string* error) {
  try {
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size()) {
      *error = "Invalid floating-point value for " + name + ": " + value;
      return false;
    }
    *output = parsed;
    return true;
  } catch (const std::exception&) {
    *error = "Invalid floating-point value for " + name + ": " + value;
    return false;
  }
}

bool NeedValue(int index, int argc, const std::string& option,
               std::string* error) {
  if (index + 1 < argc) {
    return true;
  }
  *error = "Missing value after " + option;
  return false;
}

float ClampFloat(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

std::string OutputPath(const Config& config, const std::string& file_name) {
  if (config.output_dir.empty()) {
    return file_name;
  }
  const char last = config.output_dir[config.output_dir.size() - 1];
  if (last == '/' || last == '\\') {
    return config.output_dir + file_name;
  }
  return config.output_dir + "/" + file_name;
}

bool MakeDirectory(const std::string& path) {
  if (path.empty()) {
    return true;
  }
#ifdef _WIN32
  const int status = _mkdir(path.c_str());
#else
  const int status = mkdir(path.c_str(), 0755);
#endif
  return status == 0 || errno == EEXIST;
}

unsigned char ToByte(float value) {
  const float clamped = ClampFloat(value, 0.0f, 1.0f);
  return static_cast<unsigned char>(std::lround(clamped * 255.0f));
}

void ColorMap(float value, unsigned char* r, unsigned char* g,
              unsigned char* b) {
  const float t = ClampFloat(value * 2.6f, 0.0f, 1.0f);
  const float red = 0.12f + 1.65f * t - 1.45f * t * t;
  const float green = 0.04f + 0.95f * std::sin(t * kPi);
  const float blue = 0.18f + 1.35f * (1.0f - t) * t + 0.35f * t;
  *r = ToByte(red);
  *g = ToByte(green);
  *b = ToByte(blue);
}

bool ValidateConfig(const Config& config, std::string* error) {
  if (config.width < 32 || config.height < 32) {
    *error = "width and height must both be at least 32";
    return false;
  }
  if (config.steps < 1) {
    *error = "steps must be at least 1";
    return false;
  }
  if (config.output_interval < 1) {
    *error = "output-interval must be at least 1";
    return false;
  }
  if (config.block_size < 4 || config.block_size > 32) {
    *error = "block-size must be between 4 and 32";
    return false;
  }
  if (config.backend != "gpu" && config.backend != "cpu") {
    *error = "backend must be either gpu or cpu";
    return false;
  }
  if (config.diffusion_a <= 0.0f || config.diffusion_b <= 0.0f ||
      config.dt <= 0.0f) {
    *error = "diffusion and dt values must be positive";
    return false;
  }
  if (config.feed < 0.0f || config.kill < 0.0f) {
    *error = "feed and kill values must be non-negative";
    return false;
  }
  return true;
}

}  // namespace

std::string Usage(const char* program) {
  std::ostringstream out;
  out << "Usage: " << program << " [options]\n\n"
      << "CUDA Gray-Scott reaction-diffusion simulator.\n\n"
      << "Options:\n"
      << "  --backend gpu|cpu          Execution backend (default: gpu)\n"
      << "  --width N                  Simulation width (default: 1024)\n"
      << "  --height N                 Simulation height (default: 1024)\n"
      << "  --steps N                  Number of time steps (default: 5000)\n"
      << "  --output-interval N        Write proof every N steps (default: 500)\n"
      << "  --output-dir PATH          Directory for PPM frames and metrics.csv\n"
      << "  --preset NAME              coral, spots, mitosis, or waves\n"
      << "  --feed F                   Gray-Scott feed rate\n"
      << "  --kill K                   Gray-Scott kill rate\n"
      << "  --diffusion-a F            Diffusion coefficient for chemical A\n"
      << "  --diffusion-b F            Diffusion coefficient for chemical B\n"
      << "  --dt F                     Time step size\n"
      << "  --seed N                   Deterministic initial-condition seed\n"
      << "  --block-size N             CUDA block width/height, 4..32 (default: 16)\n"
      << "  --no-images                Write metrics.csv only\n"
      << "  --help                     Show this help text\n";
  return out.str();
}

bool ParseArgs(int argc, char** argv, Config* config, std::string* error) {
  if (config == nullptr || error == nullptr) {
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      *error = "help";
      return false;
    } else if (option == "--backend") {
      if (!NeedValue(i, argc, option, error)) return false;
      config->backend = argv[++i];
    } else if (option == "--width") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseInt(argv[++i], option, &config->width, error)) return false;
    } else if (option == "--height") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseInt(argv[++i], option, &config->height, error)) return false;
    } else if (option == "--steps") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseInt(argv[++i], option, &config->steps, error)) return false;
    } else if (option == "--output-interval") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseInt(argv[++i], option, &config->output_interval, error)) {
        return false;
      }
    } else if (option == "--output-dir") {
      if (!NeedValue(i, argc, option, error)) return false;
      config->output_dir = argv[++i];
    } else if (option == "--preset") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ApplyPreset(argv[++i], config, error)) return false;
    } else if (option == "--feed") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseFloat(argv[++i], option, &config->feed, error)) return false;
    } else if (option == "--kill") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseFloat(argv[++i], option, &config->kill, error)) return false;
    } else if (option == "--diffusion-a") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseFloat(argv[++i], option, &config->diffusion_a, error)) {
        return false;
      }
    } else if (option == "--diffusion-b") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseFloat(argv[++i], option, &config->diffusion_b, error)) {
        return false;
      }
    } else if (option == "--dt") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseFloat(argv[++i], option, &config->dt, error)) return false;
    } else if (option == "--seed") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseUnsigned(argv[++i], option, &config->seed, error)) return false;
    } else if (option == "--block-size") {
      if (!NeedValue(i, argc, option, error)) return false;
      if (!ParseInt(argv[++i], option, &config->block_size, error)) {
        return false;
      }
    } else if (option == "--no-images") {
      config->write_images = false;
    } else {
      *error = "Unknown option: " + option;
      return false;
    }
  }

  return ValidateConfig(*config, error);
}

bool ApplyPreset(const std::string& preset, Config* config, std::string* error) {
  config->preset = preset;
  if (preset == "coral") {
    config->feed = 0.0545f;
    config->kill = 0.0620f;
  } else if (preset == "spots") {
    config->feed = 0.0350f;
    config->kill = 0.0650f;
  } else if (preset == "mitosis") {
    config->feed = 0.0367f;
    config->kill = 0.0649f;
  } else if (preset == "waves") {
    config->feed = 0.0140f;
    config->kill = 0.0450f;
  } else {
    *error = "Unknown preset: " + preset;
    return false;
  }
  return true;
}

std::vector<Cell> BuildInitialGrid(const Config& config) {
  std::vector<Cell> cells(static_cast<size_t>(config.width) * config.height);
  std::mt19937 rng(config.seed);
  std::uniform_real_distribution<float> noise(-0.015f, 0.015f);
  std::uniform_int_distribution<int> x_dist(config.width / 8,
                                           7 * config.width / 8);
  std::uniform_int_distribution<int> y_dist(config.height / 8,
                                           7 * config.height / 8);

  for (Cell& cell : cells) {
    cell.a = ClampFloat(1.0f + noise(rng), 0.0f, 1.0f);
    cell.b = ClampFloat(0.0f + noise(rng), 0.0f, 1.0f);
  }

  const int center_x = config.width / 2;
  const int center_y = config.height / 2;
  const int radius = std::max(8, std::min(config.width, config.height) / 10);

  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      const int dx = x - center_x;
      const int dy = y - center_y;
      const float normalized =
          static_cast<float>(dx * dx + dy * dy) / (radius * radius);
      const bool in_disc = normalized < 1.0f;
      const bool wave_seed =
          config.preset == "waves" &&
          std::fabs(std::sin(static_cast<float>(x) * 0.08f)) > 0.92f &&
          std::abs(dy) < radius * 2;
      if (in_disc || wave_seed) {
        Cell& cell = cells[static_cast<size_t>(y) * config.width + x];
        cell.a = 0.50f + 0.04f * noise(rng);
        cell.b = 0.25f + 0.70f * (1.0f - std::min(normalized, 1.0f));
      }
    }
  }

  const int patch_count = 10;
  for (int patch = 0; patch < patch_count; ++patch) {
    const int patch_x = x_dist(rng);
    const int patch_y = y_dist(rng);
    const int half_size = std::max(3, radius / 5);
    for (int y = patch_y - half_size; y <= patch_y + half_size; ++y) {
      for (int x = patch_x - half_size; x <= patch_x + half_size; ++x) {
        const int wrapped_x = (x + config.width) % config.width;
        const int wrapped_y = (y + config.height) % config.height;
        Cell& cell =
            cells[static_cast<size_t>(wrapped_y) * config.width + wrapped_x];
        cell.a = 0.45f;
        cell.b = 0.85f;
      }
    }
  }

  return cells;
}

Metrics ComputeMetrics(const std::vector<Cell>& cells, int step,
                       double elapsed_ms) {
  Metrics metrics;
  metrics.step = step;
  metrics.min_b = 1.0f;
  metrics.max_b = 0.0f;
  metrics.elapsed_ms = elapsed_ms;

  double sum_a = 0.0;
  double sum_b = 0.0;
  for (const Cell& cell : cells) {
    sum_a += cell.a;
    sum_b += cell.b;
    metrics.min_b = std::min(metrics.min_b, cell.b);
    metrics.max_b = std::max(metrics.max_b, cell.b);
  }
  if (!cells.empty()) {
    metrics.mean_a = sum_a / static_cast<double>(cells.size());
    metrics.mean_b = sum_b / static_cast<double>(cells.size());
  }
  return metrics;
}

bool EnsureOutputDirectory(const std::string& output_dir, std::string* error) {
  std::string normalized = output_dir;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');

  size_t start = 0;
  if (normalized.size() > 2 && normalized[1] == ':' && normalized[2] == '/') {
    start = 3;
  }
  while (start < normalized.size() && normalized[start] == '/') {
    ++start;
  }

  for (size_t slash = normalized.find('/', start); slash != std::string::npos;
       slash = normalized.find('/', slash + 1)) {
    const std::string partial = normalized.substr(0, slash);
    if (!partial.empty() && !MakeDirectory(partial)) {
      *error = "Unable to create output directory: " + partial;
      return false;
    }
  }

  if (!MakeDirectory(normalized)) {
    *error = "Unable to create output directory: " + normalized;
    return false;
  }
  return true;
}

bool WriteFrame(const std::vector<Cell>& cells, const Config& config, int step,
                std::string* error) {
  if (!config.write_images) {
    return true;
  }
  if (cells.size() != static_cast<size_t>(config.width) * config.height) {
    *error = "frame size does not match configured dimensions";
    return false;
  }

  std::ostringstream name;
  name << "frame_" << std::setw(6) << std::setfill('0') << step << ".ppm";
  const std::string path = OutputPath(config, name.str());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "Unable to open frame for writing: " + path;
    return false;
  }

  out << "P6\n" << config.width << " " << config.height << "\n255\n";
  std::vector<unsigned char> row(static_cast<size_t>(config.width) * 3);
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      const Cell& cell = cells[static_cast<size_t>(y) * config.width + x];
      ColorMap(cell.b, &row[static_cast<size_t>(x) * 3],
               &row[static_cast<size_t>(x) * 3 + 1],
               &row[static_cast<size_t>(x) * 3 + 2]);
    }
    out.write(reinterpret_cast<const char*>(row.data()),
              static_cast<std::streamsize>(row.size()));
  }
  return true;
}

bool WriteMetricsHeader(const Config& config, std::string* error) {
  const std::string path = OutputPath(config, "metrics.csv");
  std::ofstream out(path);
  if (!out) {
    *error = "Unable to open metrics file for writing: " + path;
    return false;
  }
  out << "step,mean_a,mean_b,min_b,max_b,elapsed_ms\n";
  return true;
}

bool AppendMetricsRow(const Config& config, const Metrics& metrics,
                      std::string* error) {
  const std::string path = OutputPath(config, "metrics.csv");
  std::ofstream out(path, std::ios::app);
  if (!out) {
    *error = "Unable to open metrics file for appending: " + path;
    return false;
  }
  out << metrics.step << ',' << std::fixed << std::setprecision(8)
      << metrics.mean_a << ',' << metrics.mean_b << ',' << metrics.min_b << ','
      << metrics.max_b << ',' << std::setprecision(3) << metrics.elapsed_ms
      << '\n';
  return true;
}

}  // namespace rd
