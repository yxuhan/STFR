"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options):
    root = Path(options.save_root).resolve()
    dataset = root
    plan = Plan(PROJECT)
    variant = ""
    mesh = dataset / "final_hack.obj"
    base = dataset / variant
    common = ["--data_root", dataset, "--mesh_name", mesh.name, "--variant", variant]
    plan.python(
        "tex.gbuffer",
        "texture/render_gbuffer.py",
        [*common, "--device", "0"],
        [mesh, dataset / "transforms.json", dataset / "image"],
        [base / "uv", base / "uv_mask"],
        [base / name for name in ["uv", "uv_mask", "uv_vis", "pho_mask"]],
    )
    plan.python(
        "tex.optimize",
        "texture/build_texture.py",
        [*common, "--device", options.device],
        [mesh, base / "uv", base / "uv_mask", dataset / "transforms.json", dataset / "image"],
        [base / "texture/image/00150/uv.png"],
        [base / "texture"],
    )
    if options.to_metahuman:
        args = ["--data_root", dataset]
        plan.python(
            "export.metahuman",
            "metahuman.py",
            args,
            [mesh, PROJECT / "misc/metahuman_transform.json"],
            [
                mesh.with_stem(mesh.stem + "_metahuman"),
                mesh.with_name(mesh.stem + "_metahuman.transform.json"),
            ],
        )
    for step in plan.steps:
        if step.name in ("tex.gbuffer", "tex.optimize"):
            step.code_inputs = [PROJECT / "texture/mesh_renderer.py"]
        if step.name == "tex.optimize":
            step.code_inputs.append(PROJECT / "texture/module.py")
    return plan.steps
