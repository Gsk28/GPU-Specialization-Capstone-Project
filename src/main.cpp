#include "reaction_diffusion.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  rd::Config config;
  std::string error;
  if (!rd::ParseArgs(argc, argv, &config, &error)) {
    if (error != "help") {
      std::cerr << "Error: " << error << "\n\n";
    }
    std::cerr << rd::Usage(argv[0]);
    return error == "help" ? 0 : 1;
  }

  bool ok = false;
  if (config.backend == "cpu") {
    ok = rd::RunCpuSimulation(config, &std::cout, &error);
  } else {
#ifdef RD_ENABLE_CUDA
    ok = rd::RunGpuSimulation(config, &std::cout, &error);
#else
    error =
        "GPU backend was requested, but this binary was built without CUDA. "
        "Build with `make gpu` or run this binary with `--backend cpu`.";
#endif
  }

  if (!ok) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  return 0;
}
