"""This stage owns its execution plan; it never imports another stage."""

from pathlib import Path
from pipeline_runtime import Plan

PROJECT = Path(__file__).resolve().parents[1]


def build_steps(options, colmap_only=False):
    root = Path(options.save_root).resolve()
    sections = ["recon"]
    if colmap_only:
        sections = ["colmap"]
    raw, mask, images = root / "raw_frames", root / "mask", root / "images"
    plan = Plan(PROJECT)
    if "recon" in sections:
        plan.python(
            "recon.input_format",
            "reconstruction/to_2dgs_format.py",
            ["--data_root", root],
            [raw, mask],
            [images],
            [images],
        )
    if "recon" in sections or "colmap" in sections:
        db, sparse = root / "database.db", root / "sparse"
        model = sparse / "0"
        bins = [model / name for name in ["cameras.bin", "images.bin", "points3D.bin"]]
        plan.command(
            "recon.colmap.features",
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                images,
                "--ImageReader.camera_model",
                "PINHOLE",
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.max_image_size",
                "4000",
                "--FeatureExtraction.use_gpu",
                "1",
                "--ImageReader.mask_path",
                mask,
            ],
            [images, mask],
            [db],
            [db],
            fresh=False,
        )
        plan.command(
            "recon.colmap.match",
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                db,
                "--FeatureMatching.guided_matching",
                "1",
                "--FeatureMatching.use_gpu",
                "1",
            ],
            [db],
            [db],
            fresh=False,
        )
        plan.command(
            "recon.colmap.map",
            [
                "colmap",
                "mapper",
                "--database_path",
                db,
                "--image_path",
                images,
                "--output_path",
                sparse,
            ],
            [db, images],
            bins,
            [sparse],
            mkdir=[sparse],
        )
        # COLMAP's mapper touches the input SQLite database during successful reads.
        # Its content is validated by the preceding feature/matching output records.
        plan.steps[-1].mutable_inputs = [db]
        plan.command(
            "recon.colmap.bundle_adjust",
            [
                "colmap",
                "bundle_adjuster",
                "--input_path",
                model,
                "--output_path",
                model,
                "--BundleAdjustment.max_num_iterations",
                "100",
            ],
            [db, *bins],
            bins,
        )
        plan.command(
            "recon.colmap.text_model",
            [
                "colmap",
                "model_converter",
                "--input_path",
                model,
                "--output_path",
                model,
                "--output_type",
                "TXT",
            ],
            bins,
            [model / name for name in ["cameras.txt", "images.txt", "points3D.txt"]],
        )
    if "recon" in sections:
        recon = root / "recon"
        checkpoint = recon / "point_cloud/iteration_30000/point_cloud.ply"
        plan.python(
            "recon.2dgs.train",
            "reconstruction/2d-gaussian-splatting/train.py",
            ["-s", root, "-m", recon, "--port", options.gs_port],
            [
                images,
                root / "sparse/0/images.bin",
                root / "sparse/0/cameras.bin",
                root / "sparse/0/points3D.bin",
            ],
            [checkpoint, recon / "cameras.json"],
            [recon],
        )
        plan.python(
            "recon.2dgs.mesh",
            "reconstruction/2d-gaussian-splatting/render.py",
            ["-s", root, "-m", recon, "--mesh_res", "1024"],
            [checkpoint],
            [recon / "train/ours_30000/fuse_post.ply"],
        )
        plan.python(
            "recon.geometry",
            "reconstruction/to_my_format.py",
            ["--data_root", root],
            [recon / "train/ours_30000/fuse_post.ply", recon / "cameras.json"],
            [root / "2dgs_recon.obj", root / "transforms.json"],
        )
    for step in plan.steps:
        if step.name.startswith("recon.2dgs."):
            for directory in ["utils", "scene", "gaussian_renderer", "arguments"]:
                step.code_inputs.extend(
                    (PROJECT / "reconstruction/2d-gaussian-splatting" / directory).rglob("*.py")
                )
    return plan.steps
