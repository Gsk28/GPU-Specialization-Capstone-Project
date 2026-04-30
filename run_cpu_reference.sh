#!/usr/bin/env bash
set -euo pipefail

make cpu
mkdir -p proof/cpu_reference
./build/rd_cpu \
  --backend cpu \
  --width 256 \
  --height 256 \
  --steps 800 \
  --output-interval 200 \
  --preset coral \
  --output-dir proof/cpu_reference \
  2>&1 | tee proof/cpu_reference/run.log
