"""Convert generated PPM proof frames to PNG previews.

This helper is optional. The simulator writes dependency-free PPM files so the
core CUDA project does not need an image library. Use this script when you want
PNG files for a README, slide deck, or video editor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    for ppm_path in sorted(args.input_dir.glob("frame_*.ppm")):
        png_path = ppm_path.with_suffix(".png")
        with Image.open(ppm_path) as image:
            image.save(png_path)
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
