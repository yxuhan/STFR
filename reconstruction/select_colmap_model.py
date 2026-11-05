"""Select the COLMAP reconstruction with the most registered images."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for model in args.sparse_dir.iterdir():
        image_file = model / "images.bin"
        if not model.is_dir() or not model.name.isdecimal() or not image_file.is_file():
            continue
        with image_file.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid COLMAP images file: {image_file}")
        count = struct.unpack("<Q", header)[0]
        if count:
            candidates.append((count, int(model.name), model))
    if not candidates:
        raise ValueError(f"No COLMAP models with registered images in {args.sparse_dir}")

    # Prefer the lowest model index on a tie, keeping model 0 when possible.
    count, index, selected = max(candidates, key=lambda item: (item[0], -item[1]))
    zero = args.sparse_dir / "0"
    args.report_dir.mkdir(parents=True, exist_ok=False)
    if index != 0:
        temporary = args.sparse_dir / f".selected_{uuid.uuid4().hex}"
        backup = args.report_dir / "original_model_0"
        shutil.copytree(selected, temporary)
        if zero.exists():
            os.replace(zero, backup)
        try:
            os.replace(temporary, zero)
        except BaseException:
            if backup.exists() and not zero.exists():
                os.replace(backup, zero)
            raise

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (zero / name).is_file():
            raise FileNotFoundError(f"Selected COLMAP model lacks {name}: {zero}")
    report = {
        "selected_original_index": index,
        "registered_images": count,
        "candidates": [
            {"model_index": number, "registered_images": size}
            for size, number, _ in sorted(candidates, key=lambda item: item[1])
        ],
    }
    (args.report_dir / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected COLMAP model {index} with {count} registered images", flush=True)


if __name__ == "__main__":
    main()
