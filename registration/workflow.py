"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    (raw, ref, reg) = (root / "raw_frames", root / "refinement", root / "register")
    (sample, volume) = (ref / "sample", ref / "texture")
    (coarse, fine, wrap) = (reg / "coarse_align", reg / "fine_align", reg / "wrap")
    dataset = root / "sample_dataset"
    plan = Plan(PROJECT)

    def add_appearance_step():
        plan.python(
            "reg.appearance",
            "registration/volume_texture.py",
            [
                "--mesh_path",
                root / "2dgs_recon.obj",
                "--ckpt_path",
                volume / "latest.pth",
                "--normalization",
                sample / "normalization.json",
                "--output",
                reg / "input_appearance.json",
            ],
            [root / "2dgs_recon.obj", volume / "latest.pth"]
            + ([sample / "normalization.json"] if (sample / "normalization.json").exists() else []),
            [reg / "input_appearance.json"],
        )

    add_appearance_step()
    plan.python(
        "reg.coarse_align",
        "registration/align/align_coarse.py",
        [
            "--img_root",
            raw,
            "--mesh_path",
            root / "2dgs_recon.obj",
            "--cam_path",
            root / "transforms.json",
            "--save_root",
            coarse,
            "--appearance_path",
            reg / "input_appearance.json",
        ],
        [raw, root / "2dgs_recon.obj", root / "transforms.json", reg / "input_appearance.json"],
        [
            coarse / "align_init.pkl",
            coarse / "align_scaled.ply",
            coarse / "align_scaled.json",
            coarse / "appearance.json",
        ],
        [coarse],
        cwd=PROJECT / "registration",
    )
    plan.python(
        "reg.flame_fit",
        "registration/align/AlbedoMMFitting/fitting.py",
        ["--img_root", raw, "--coarse_fitting_root", coarse, "--save_root", fine],
        [raw, coarse / "align_init.pkl", coarse / "align_scaled.json"],
        [fine / "fitting.pkl"],
        [fine],
    )
    plan.python(
        "reg.canonical_space",
        "registration/align/AlbedoMMFitting/to_canonical.py",
        [
            "--coarse_fitting_root",
            coarse,
            "--save_root",
            fine,
            "--appearance_path",
            coarse / "appearance.json",
        ],
        [
            coarse / "appearance.json",
            coarse / "align_scaled.ply",
            coarse / "align_scaled.json",
            fine / "fitting.pkl",
        ],
        [fine / "align_canonical.ply", fine / "align_canonical.json", fine / "appearance.json"],
    )
    plan.python(
        "reg.frontal_landmarks",
        "registration/to_wrap.py",
        [
            "--data_root",
            fine,
            "--save_root",
            wrap,
            "--ldm_type",
            options.reg_ldm_type,
            "--reg_template",
            options.reg_template,
            "--appearance_path",
            fine / "appearance.json",
        ],
        [fine / "appearance.json", fine / "align_canonical.ply",
         PROJECT / "registration/assets" / (options.reg_template + "_eye_template.obj")]
        + ([PROJECT / "registration/assets/fill_eye_landmarks.json"] if options.reg_template == "fill" else []),
        [
            wrap / "wrap.wrap",
            wrap / "tgt.json",
            wrap / "bfm.json",
            wrap / "frontal_rgb.png",
            wrap / "frontal_xyz.pt",
            wrap / "target_geometry.obj",
        ],
        [wrap],
    )
    plan.command(
        "reg.wrap",
        ["xvfb-run", "--auto-servernum", "./WrapCmd", "compute", wrap / "wrap.wrap"],
        [wrap / "wrap.wrap", wrap / "tgt.json", wrap / "bfm.json", wrap / "target_geometry.obj"],
        [wrap / "final_hack.obj"],
        cwd=PROJECT / "registration/wrap",
    )
    plan.python(
        "reg.texture_dataset",
        "registration/build_dataset.py",
        ["--data_root", root],
        [wrap / "final_hack.obj", fine / "align_canonical.json", sample / "image"],
        [dataset / "final_hack.obj", dataset / "transforms.json", dataset / "image"],
        [dataset],
    )
    dependencies = {
        "reg.coarse_align": [
            "registration/volume_texture.py",
            "registration/utils/mesh_renderer.py",
        ],
        "reg.flame_fit": [
            "registration/align/AlbedoMMFitting/config.py",
            "registration/align/AlbedoMMFitting/util.py",
            "registration/align/AlbedoMMFitting/face_detector.py",
            "registration/align/AlbedoMMFitting/mesh_renderer.py",
            "registration/align/AlbedoMMFitting/render_utils.py",
        ],
        "reg.canonical_space": [
            "registration/volume_texture.py",
            "registration/align/AlbedoMMFitting/config.py",
        ],
        "reg.frontal_landmarks": [
            "registration/volume_texture.py",
            "registration/utils/mesh_renderer.py",
        ],
    }
    for step in plan.steps:
        step.code_inputs = [PROJECT / path for path in dependencies.get(step.name, [])]
        if step.name in ("reg.flame_fit", "reg.canonical_space"):
            step.code_inputs.extend(
                (PROJECT / "registration/align/AlbedoMMFitting/flame").rglob("*.py")
            )
        if step.name == "reg.frontal_landmarks":
            step.code_inputs.extend((PROJECT / "registration/eyelid_detector").rglob("*.py"))
    return plan.steps
