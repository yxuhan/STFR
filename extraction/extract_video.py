"""Extract evenly spaced video frames within a maximum image dimension."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image


def frame_count(video, *, decode=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    field = "nb_read_frames" if decode else "nb_frames"
    if decode:
        command.append("-count_frames")
    command += ["-show_entries", f"stream={field}", "-of", "json", str(video)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    value = streams[0].get(field) if streams else None
    return int(value) if value and str(value).isdigit() and int(value) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num_frame", type=int, default=80)
    parser.add_argument("--max_image_size", type=int, default=960)
    args = parser.parse_args()
    if args.num_frame < 1 or args.max_image_size < 1:
        parser.error("num_frame and max_image_size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.output_dir}")

    total = frame_count(args.video_path) or frame_count(args.video_path, decode=True)
    if not total:
        raise ValueError(f"Cannot determine video frame count: {args.video_path}")

    def extract(count, directory):
        target = min(args.num_frame, count)
        # A fractional interval selects exactly target frames without repeating
        # images when the video length is not divisible by the requested count.
        select = f"not(eq(floor(n*{target}/{count}),floor((n-1)*{target}/{count})))"
        scale = (
            f"scale=w='trunc(iw*min(1,{args.max_image_size}/max(iw,ih)))':"
            f"h='trunc(ih*min(1,{args.max_image_size}/max(iw,ih)))'"
        )
        command = ["ffmpeg", "-nostdin", "-y", "-i", str(args.video_path),
                   "-vf", f"select='{select}',{scale},setsar=1", "-vsync", "vfr",
                   "-q:v", "1", str(directory / "%05d.png")]
        print(f"Video frames: {count}; requested: {target}; interval: {count / target:.3f}", flush=True)
        subprocess.run(command, check=True)
        return target, sorted(directory.glob("*.png"))

    with tempfile.TemporaryDirectory(prefix=".stfr_extract_", dir=args.output_dir.parent) as temp:
        directory = Path(temp)
        target, frames = extract(total, directory)
        if len(frames) != target:
            actual = frame_count(args.video_path, decode=True)
            if not actual or actual == total:
                raise ValueError(f"Expected {target} frames but extracted {len(frames)}")
            print(f"Video metadata reported {total} frames; decoded count is {actual}. Retrying.", flush=True)
            for path in frames:
                path.unlink()
            total = actual
            target, frames = extract(total, directory)
            if len(frames) != target:
                raise ValueError(f"Expected {target} frames but extracted {len(frames)}")
        with Image.open(frames[0]) as image:
            width, height = image.size
        if max(width, height) > args.max_image_size:
            raise ValueError(f"Extracted image exceeds maximum size: {width}x{height}")
        for path in frames:
            shutil.move(str(path), args.output_dir / path.name)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "video_path": str(args.video_path.resolve()), "video_frames": total,
        "num_frame_requested": args.num_frame, "num_frame_extracted": target,
        "average_frame_interval": total / target, "max_image_size": args.max_image_size,
        "output_width": width, "output_height": height,
    }, indent=2) + "\n")
    print(f"Extracted {target} frames at {width}x{height} into {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
