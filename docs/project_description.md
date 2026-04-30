# Project Description

## Goal

The goal of this capstone is to demonstrate a GPU-accelerated simulation that
is visual, measurable, and directly tied to common GPU programming concepts. I
chose the Gray-Scott reaction-diffusion model because it has a compact
mathematical definition but creates surprisingly complex images. It is also a
good fit for CUDA because every cell in the grid can be updated in parallel.

## Algorithm

The simulation tracks two chemicals, A and B, on a 2D grid. Each cell evolves
according to diffusion, reaction, feed, and kill terms:

```text
A' = A + (Da * Laplacian(A) - A * B * B + feed * (1 - A)) * dt
B' = B + (Db * Laplacian(B) + A * B * B - (kill + feed) * B) * dt
```

The Laplacian uses a nine-point stencil:

```text
diagonal neighbors: 0.05
cardinal neighbors: 0.20
center:            -1.00
```

The output images colorize chemical B, so changes in the reaction field are
visible across the generated frames.

## CUDA Implementation

The CUDA implementation is in `src/gpu_backend.cu`.

Important pieces:

- `UpdateKernel` launches a 2D grid of thread blocks.
- Each CUDA thread handles one cell of the simulation image.
- The kernel reads from `device_current` and writes to `device_next`.
- Host code swaps the two device pointers after each step.
- Periodically, the current device buffer is copied back to the host so the
  program can write a PPM frame and append metrics to `metrics.csv`.
- CUDA events measure elapsed GPU time.

This avoids inter-thread synchronization inside the stencil update because each
thread writes to a unique output cell.

## CLI and Support Files

The program accepts command-line arguments for backend, dimensions, number of
steps, output interval, output directory, preset, random seed, and CUDA block
size. The repository also includes:

- `Makefile` for CPU and GPU builds
- `run.sh` for the main CUDA demonstration
- `run_cpu_reference.sh` for non-CUDA validation
- `run.ps1` for Windows users

## Proof Artifacts

The proof artifacts are written into the `proof` directory:

- `frame_*.ppm`: rendered simulation images at selected steps
- `metrics.csv`: summary statistics for the concentration fields
- `run.log`: command output showing backend, grid, preset, and status

The included `proof/cpu_reference` artifacts were generated locally on a
machine without CUDA. The final submission should add `proof/gpu_demo` from a
CUDA machine so reviewers can see the device name and GPU execution log.

## Challenges and Lessons Learned

The main programming challenge is respecting the data dependency in a stencil
simulation. If each thread updated the same array it read from, neighboring
threads could observe partially updated state. The double-buffered design fixes
that by separating the read state from the write state.

Another useful lesson is that GPU acceleration is not just about writing a
kernel. The host code still matters: command-line configuration, output
artifacts, error checking, and reproducible seeds make the project easier to
run and review.

## Possible Next Steps

- Add shared-memory tiling to reduce repeated global memory reads.
- Compare GPU timing against CPU timing for several grid sizes.
- Write frames as PNG using a small image library.
- Generate a short video from the output frames.
- Add more presets and a mode for loading an external seed image.
