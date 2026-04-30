# CUDA Gray-Scott Reaction-Diffusion Capstone

This project is a CUDA/C++ capstone for the GPU Specialization final
assignment. It simulates a Gray-Scott reaction-diffusion system, a simple
chemical model that produces organic-looking evolving patterns from two
concentration fields. Each time step updates every pixel independently from a
small neighborhood, so it maps naturally to a CUDA 2D grid of GPU threads.

The GPU backend is the intended capstone implementation. A CPU backend is also
included for validation and for machines without CUDA; it is not the main
assignment path.

## Why This Uses the GPU

The hot loop is `UpdateKernel` in `src/gpu_backend.cu`. For each time step, a
CUDA thread updates one cell of the simulation grid:

- reads the eight neighboring cells plus the center cell
- computes weighted Laplacians for chemicals A and B
- applies the Gray-Scott reaction equations
- writes the next state into a second GPU buffer

The simulation uses ping-pong device buffers (`device_current` and
`device_next`) so each step can run in parallel without threads overwriting
neighbor data that other threads still need.

## Repository Layout

```text
.
|-- Makefile
|-- README.md
|-- SUBMISSION.md
|-- bin/
|-- data/
|-- lib/
|-- run.sh
|-- run.ps1
|-- run_cpu_reference.sh
|-- scripts/
|   `-- convert_ppm_to_png.py
|-- docs/
|   |-- project_description.md
|   `-- presentation_outline.md
|-- proof/
|   `-- cpu_reference/
|       |-- frame_000000.ppm
|       |-- frame_000200.ppm
|       |-- frame_000400.ppm
|       |-- frame_000600.ppm
|       |-- frame_000800.ppm
|       |-- frame_*.png
|       |-- metrics.csv
|       `-- run.log
`-- src/
    |-- common.cpp
    |-- cpu_backend.cpp
    |-- gpu_backend.cu
    |-- main.cpp
    `-- reaction_diffusion.h
```

## Build Requirements

Primary GPU build:

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- GNU Make or compatible `make`
- C++17-capable host compiler

Reference CPU build:

- C++17-capable compiler such as `g++`
- GNU Make or compatible `make`

## Build and Run

On a CUDA machine:

```bash
make gpu
./build/rd_cuda --backend gpu --width 1024 --height 1024 --steps 5000 \
  --output-interval 500 --preset coral --block-size 16 \
  --output-dir proof/gpu_demo
```

Or use the convenience script:

```bash
./run.sh
```

For a non-CUDA reference run:

```bash
make cpu
./build/rd_cpu --backend cpu --width 256 --height 256 --steps 800 \
  --output-interval 200 --preset coral --output-dir proof/cpu_reference
```

On Windows with MinGW:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Cpu
```

## Command-Line Options

```text
--backend gpu|cpu          Execution backend
--width N                  Simulation width
--height N                 Simulation height
--steps N                  Number of time steps
--output-interval N        Write proof every N steps
--output-dir PATH          Directory for PPM frames and metrics.csv
--preset NAME              coral, spots, mitosis, or waves
--feed F                   Gray-Scott feed rate
--kill K                   Gray-Scott kill rate
--diffusion-a F            Diffusion coefficient for chemical A
--diffusion-b F            Diffusion coefficient for chemical B
--dt F                     Time step size
--seed N                   Deterministic initial-condition seed
--block-size N             CUDA block width/height, 4..32
--no-images                Write metrics.csv only
--help                     Show help
```

## Proof of Execution

The `proof/cpu_reference` folder contains a local reference run created on a
machine without CUDA:

```text
backend=cpu
grid=256x256
steps=800
preset=coral
status=ok
```

The generated `metrics.csv` shows the concentration field changing over time,
and the `frame_*.ppm` files show the evolving output images.

Optional PNG previews can be created with Pillow:

```bash
python -m pip install pillow
python scripts/convert_ppm_to_png.py proof/gpu_demo
```

For the final course submission, run `./run.sh` in a CUDA-enabled environment
and commit the generated `proof/gpu_demo` folder. The GPU log will include the
CUDA device name and compute capability, which makes GPU execution clear to a
peer reviewer.

## Project Summary

This project demonstrates:

- mapping a 2D numerical stencil to CUDA thread blocks
- avoiding read/write hazards with double-buffered GPU memory
- using CUDA events to collect elapsed GPU timing
- copying periodic output back to the host for proof images and CSV metrics
- exposing useful CLI controls for grid size, step count, preset, output path,
  and CUDA block size

More detail is in `docs/project_description.md`.
