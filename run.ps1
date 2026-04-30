param(
  [switch]$Cpu
)

if ($Cpu) {
  mingw32-make cpu
  New-Item -ItemType Directory -Force -Path proof\cpu_reference | Out-Null
  .\build\rd_cpu.exe `
    --backend cpu `
    --width 256 `
    --height 256 `
    --steps 800 `
    --output-interval 200 `
    --preset coral `
    --output-dir proof\cpu_reference `
    *>&1 | Tee-Object -FilePath proof\cpu_reference\run.log
} else {
  mingw32-make gpu
  New-Item -ItemType Directory -Force -Path proof\gpu_demo | Out-Null
  .\build\rd_cuda.exe `
    --backend gpu `
    --width 1024 `
    --height 1024 `
    --steps 5000 `
    --output-interval 500 `
    --preset coral `
    --block-size 16 `
    --output-dir proof\gpu_demo `
    *>&1 | Tee-Object -FilePath proof\gpu_demo\run.log
}
