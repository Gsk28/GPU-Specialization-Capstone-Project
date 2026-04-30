# Proof Artifacts

`cpu_reference` contains a deterministic reference run generated on a local
machine without CUDA:

```bash
./build/rd_cpu --backend cpu --width 256 --height 256 --steps 800 \
  --output-interval 200 --preset coral --output-dir proof/cpu_reference
```

For the final course submission, run the CUDA path in a GPU-enabled lab:

```bash
./run.sh
```

Then commit the resulting `proof/gpu_demo` directory. The GPU run log should
show `backend=gpu`, the CUDA device name, and `status=ok`.

The PNG files in `cpu_reference` were created from the simulator's PPM frames
with `scripts/convert_ppm_to_png.py` so the images are easy to preview.
