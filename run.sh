#!/usr/bin/env bash
set -euo pipefail

make gpu
mkdir -p proof/gpu_demo
./build/rd_cuda \
  --backend gpu \
  --width 1024 \
  --height 1024 \
  --steps 5000 \
  --output-interval 500 \
  --preset coral \
  --block-size 16 \
  --output-dir proof/gpu_demo \
  2>&1 | tee proof/gpu_demo/run.log
