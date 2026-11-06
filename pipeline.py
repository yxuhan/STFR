"""Root orchestration through independent stage CLIs and their JSON plan protocol."""

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timezone

from pipeline_runtime import Runner, StageError, Step, atomic_json, Plan

PROJECT = Path(__file__).resolve().parent
PYTHON = sys.executable
ORDER = ["extract", "mat", "recon", "refine", "reg", "tex", "export"]


def build_plan(options, sections):
    root = Path(options.save_root).resolve()
    specs = {
        "extract": (
            "extraction/run_extraction.py",
            [
                "--video_path",
                str(Path(options.video_path).resolve()),
                "--num_frame",
                str(options.num_frame),
                "--max_image_size",
                str(options.max_image_size),
            ],
        ),
        "mat": ("matting/run_stage.py", []),
        "recon": ("reconstruction/run_reconstruction.py", ["--gs_port", str(options.gs_port)]),
        "refine": ("refinement/run_refinement.py", []),
        "reg": (
            "registration/run_registration.py",
            ["--reg_template", options.reg_template, "--ldm_type", options.reg_ldm_type],
        ),
        "tex": ("texture/run_texture.py", []),
    }
    steps = []
    for stage in sections:
        if stage == "export":
            continue
        script, extra = specs[stage]
        dataset = root / "sample_dataset" if stage == "tex" else root
        command = [
            PYTHON,
            str(PROJECT / script),
            "--data_root",
            str(dataset),
            "--device",
            options.device,
            "--plan_json",
            *extra,
        ]
        result = subprocess.run(command, cwd=PROJECT, check=False, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Cannot plan stage {stage}: {result.stderr}")
        for item in json.loads(result.stdout)["steps"]:
            item["cwd"] = Path(item["cwd"])
            for field in ["inputs", "outputs", "reset", "mkdir", "code_inputs", "mutable_inputs"]:
                item[field] = list(map(Path, item[field]))
            steps.append(Step(**item))
    if options.to_metahuman:
        dataset = root / "sample_dataset"
        name = "final_hack"
        plan = Plan(PROJECT)
        plan.python(
            "export.metahuman",
            "metahuman.py",
            ["--data_root", dataset],
            [dataset / (name + ".obj"), PROJECT / "misc/metahuman_transform.json"],
            [dataset / (name + "_metahuman.obj"), dataset / (name + "_metahuman.transform.json")],
        )
        steps.extend(plan.steps)
    return steps


def main():
    parser = argparse.ArgumentParser(description="STFR checked stage pipeline")
    parser.add_argument("--video_path", default="workspace/0227.mp4")
    parser.add_argument("--num_frame", type=int, default=80)
    parser.add_argument("--max_image_size", type=int, default=960)
    parser.add_argument("--save_root", "--data_root", dest="save_root", default="workspace/0227")
    parser.add_argument("--func", default="extract-mat-recon-refine-reg-tex")
    parser.add_argument("--reg_template", choices=["close", "open", "fill"], default="open")
    parser.add_argument(
        "--reg_ldm_type",
        "--ldm_type",
        dest="reg_ldm_type",
        choices=["ibug", "eyelid", "interp"],
        default="eyelid",
    )
    parser.add_argument("--to_metahuman", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--gs_port", type=int, default=6010)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated successful steps from the previous invocation",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the complete stage plan without running or writing artifacts",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current/last invocation without running any steps",
    )
    options = parser.parse_args()
    options.save_root = str(Path(options.save_root).resolve())
    if options.status:
        path = Path(options.save_root) / "pipeline_status.json"
        if not path.is_file():
            parser.error(f"No pipeline status at {path}")
        state = json.loads(path.read_text())
        print(
            f"{state['status'].upper()} | run {state['run_id']} | current: {state['current_step']}"
        )
        for entry in state["steps"]:
            elapsed = entry.get("elapsed_seconds", 0)
            if entry["status"] == "running" and entry.get("started"):
                elapsed = (
                    datetime.now(timezone.utc) - datetime.fromisoformat(entry["started"])
                ).total_seconds()
            print(f"{entry['number']:02d} {entry['status']:10s} {entry['name']} {elapsed:.1f}s")
        print("Logs: " + state["log_dir"])
        return 0
    if "," in options.device or not options.device.isdigit():
        parser.error("--device must select one physical CUDA device")
    requested = options.func.split("-")
    if set(requested) - set(ORDER) or len(set(requested)) != len(requested):
        parser.error("--func must contain unique stage names from " + ", ".join(ORDER))
    if "export" in requested and not options.to_metahuman:
        # Preserve the prior no-op export/default-off behavior.
        requested = [item for item in requested if item != "export"]
    sections = [item for item in ORDER if item in requested]
    steps = build_plan(options, sections)
    if options.dry_run:
        for number, step in enumerate(steps, 1):
            print(
                f"[{number:02d}/{len(steps):02d}] {step.name}\n  cwd={step.cwd}\n  argv={list(map(str, step.command))}"
            )
        return 0
    try:
        state = Runner(
            options.save_root, PROJECT, options.resume, env={"CUDA_VISIBLE_DEVICES": options.device}
        ).run(steps, vars(options))
    except KeyboardInterrupt:
        return 130
    except (StageError, ValueError) as error:
        print(str(error), file=sys.stderr)
        atomic_json(
            Path(options.save_root) / "outputs.json",
            dict(
                status="failed",
                error=str(error),
                pipeline_status=str(Path(options.save_root) / "pipeline_status.json"),
            ),
        )
        return 1
    dataset = Path(options.save_root) / "sample_dataset"
    name = "final_hack"
    variant = ""
    available = lambda path: str(path) if path.is_file() else None
    atomic_json(
        Path(options.save_root) / "outputs.json",
        dict(
            run_id=state["run_id"],
            status=state["status"],
            mesh=available(dataset / (name + ".obj")),
            texture=available(dataset / variant / "texture/image/00150/uv.png"),
            metahuman_mesh=available(dataset / (name + "_metahuman.obj"))
            if options.to_metahuman
            else None,
            appearance=available(Path(options.save_root) / "refinement/appearance.json"),
            logs=state["log_dir"],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
