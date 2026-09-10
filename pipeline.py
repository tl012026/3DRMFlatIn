"""
Script role:
    CLI / entry script to launch the pipeline framework.
    Parses args, loads configs/default.yaml (or a user config),
    and calls project.pipeline.run_pipeline.
    No algorithm logic here.

Inputs:
    - --config: path to YAML config (default: configs/default.yaml)
    - --input_dir or --image_paths: multi-view image source
    - --output_dir: output directory
    (CLI values override config keys when provided)

Outputs:
    - pipeline_result from run_pipeline
    - files written under output_dir by the pipeline stages

Variables (contract):
    - args: parsed CLI namespace
    - config: loaded YAML dict / object
    - image_paths: List[str] resolved from args/config
    - output_dir: str | Path
    - pipeline_result: return value of run_pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "ultralytics", ROOT / "vggt-omega", ROOT / "sam2"):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

import yaml

from project.pipeline import run_pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    """
    Inputs:
        CLI args + config file
    Outputs:
        runs pipeline; exits with status code later
    Variables used:
        args, config, image_paths, output_dir, pipeline_result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "default.yaml"))
    parser.add_argument("--input_dir")
    parser.add_argument("--image_paths", nargs="*")
    parser.add_argument("--output_dir")
    parser.add_argument("--device")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device

    image_paths = list(args.image_paths or config.get("image_paths") or [])
    input_dir = args.input_dir or config.get("input_dir")
    if not image_paths:
        if not input_dir:
            raise SystemExit("provide --input_dir or --image_paths")
        image_paths = sorted(
            str(p) for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            raise SystemExit(f"no images in {input_dir}")

    output_dir = args.output_dir or config["output_dir"]
    pipeline_result = run_pipeline(config, image_paths, output_dir)
    return pipeline_result


if __name__ == "__main__":
    main()
