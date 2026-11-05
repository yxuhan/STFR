"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    (raw, ref) = (root / "raw_frames", root / "refinement")
    (sample, volume) = (ref / "sample", ref / "texture")
    plan = Plan(PROJECT)

    def add_appearance_step():
        plan.python(
            "refine.appearance",
            "refinement/volume_texture.py",
            [
                "--mesh_path",
                root / "2dgs_recon.obj",
                "--ckpt_path",
                volume / "latest.pth",
                "--normalization",
                sample / "normalization.json",
                "--output",
                ref / "appearance.json",
            ],
            [root / "2dgs_recon.obj", volume / "latest.pth"] + [sample / "normalization.json"],
            [ref / "appearance.json"],
        )

    plan.python(
        "refine.sharpness",
        "refinement/select_frame/compute_sharpness.py",
        ["--img_root", raw, "--save_root", sample],
        [raw],
        [sample / "sharpness.pkl"],
        [sample],
    )
    plan.python(
        "refine.select_views",
        "refinement/select_frame/sample_by_sharpness.py",
        [
            "--img_root",
            raw,
            "--cam_path",
            root / "transforms.json",
            "--save_root",
            sample,
            "--num_view",
            "16",
        ],
        [raw, sample / "sharpness.pkl", root / "transforms.json"],
        [sample / "select_sharp.json", sample / "image"],
        [sample / "image"],
    )
    plan.python(
        "refine.xyz_maps",
        "refinement/render_position_map.py",
        [
            "--mesh_path",
            root / "2dgs_recon.obj",
            "--cam_path",
            sample / "select_sharp.json",
            "--img_root",
            raw,
            "--save_root",
            sample,
        ],
        [root / "2dgs_recon.obj", sample / "select_sharp.json", raw],
        [sample / "pointmap", sample / "pointmap_mask", sample / "normalization.json"],
        [sample / "pointmap", sample / "pointmap_mask", sample / "pointmap_vis"],
    )
    plan.python(
        "refine.volume_texture",
        "refinement/build_texture.py",
        [
            "--img_root",
            sample / "image",
            "--pointmap_root",
            sample / "pointmap",
            "--mask_root",
            sample / "pointmap_mask",
            "--save_root",
            volume,
        ],
        [sample / "image", sample / "pointmap", sample / "pointmap_mask"],
        [volume / "latest.pth"],
        [volume],
    )
    add_appearance_step()
    dependencies = {
        "refine.xyz_maps": ["refinement/mesh_renderer.py"],
        "refine.volume_texture": ["refinement/network.py"],
    }
    for step in plan.steps:
        step.code_inputs = [PROJECT / path for path in dependencies.get(step.name, [])]
    return plan.steps
