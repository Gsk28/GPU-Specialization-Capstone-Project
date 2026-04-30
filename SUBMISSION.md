# Submission Text

Repository URL:

```text
<paste your public GitHub/GitLab/Bitbucket/Drive URL here>
```

Short project description:

```text
I built a CUDA/C++ Gray-Scott reaction-diffusion simulator. The program maps a
2D numerical stencil to CUDA thread blocks, with one GPU thread updating each
cell of the simulation grid. It uses double-buffered GPU memory to avoid
read/write hazards, CUDA events for timing, and command-line options for grid
size, number of steps, preset, output directory, random seed, and block size.
The output proof artifacts include simulation frames and a metrics.csv file
showing how the concentration fields change over time. A CPU reference backend
is included only for validation on non-CUDA machines; the main capstone run is
the CUDA backend invoked by ./run.sh.
```

Presentation/demo URL:

```text
<paste your 5-10 minute presentation video URL here>
```
