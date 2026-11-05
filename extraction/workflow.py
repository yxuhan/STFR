"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    raw = root / "raw_frames"
    plan = Plan(PROJECT)
    if options.video_step_size < 1 or options.video_ds_ratio <= 0:
        raise ValueError("video_step_size and video_ds_ratio must be positive")
    video = Path(options.video_path).resolve()
    plan.command(
        "extract.frames",
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video,
            "-vf",
            f"select='not(mod(n,{options.video_step_size}))',scale=iw*{options.video_ds_ratio}:ih*{options.video_ds_ratio},setsar=1:1",
            "-vsync",
            "vfr",
            "-q:v",
            "1",
            raw / "%05d.png",
        ],
        [video],
        [raw],
        [raw],
        mkdir=[raw],
    )
    return plan.steps
