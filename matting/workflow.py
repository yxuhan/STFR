"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    (raw, mask) = (root / "raw_frames", root / "mask")
    plan = Plan(PROJECT)
    plan.python(
        "mat.foreground",
        "matting/run_matting.py",
        ["--input_root", raw, "--output_root", mask],
        [raw],
        [mask],
        [mask],
    )
    return plan.steps
