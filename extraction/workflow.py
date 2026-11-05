"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    raw = root / "raw_frames"
    plan = Plan(PROJECT)
    if options.num_frame < 1 or options.max_image_size < 1:
        raise ValueError("num_frame and max_image_size must be positive")
    video = Path(options.video_path).resolve()
    report = root / "extraction.json"
    plan.python(
        "extract.frames",
        "extraction/extract_video.py",
        ["--video_path", video, "--output_dir", raw, "--report", report,
         "--num_frame", options.num_frame, "--max_image_size", options.max_image_size],
        [video],
        [raw, report],
        [raw, report],
        mkdir=[raw],
    )
    return plan.steps
