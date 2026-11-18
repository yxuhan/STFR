"""Initialize FLAME from multiview landmarks and fit it in one pass."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d
from scipy.optimize import least_squares

from config import parse_args
from face_detector import FaceDetector
from flame.FLAME import FLAME

STABLE_IDS = (2, 4, 5, 6, 33, 98, 133, 168, 195, 197, 263, 327, 362)
MAX_VIEWS = 10
SAMPLE_SEED = 42
ROBUST_SCALE_PX = 4.0


def load_views(config, embedding_ids):
    meta_path = Path(config.coarse_fitting_root) / "align_scaled.json"
    meta = json.loads(meta_path.read_text())
    intrinsic = np.array([[meta["fl_x"], 0, meta["cx"]],
                          [0, meta["fl_y"], meta["cy"]],
                          [0, 0, 1]])
    detector = FaceDetector("google")
    names, projections, cameras, observations = [], [], [], []
    frames = meta["frames"]
    for frame in frames:
        name = Path(frame["file_path"]).name
        image = cv2.imread(str(Path(config.img_root) / name))
        if image is None:
            continue
        landmarks = detector.dense(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if landmarks is None:
            continue
        landmarks = np.asarray(landmarks).reshape(-1, 2)
        if len(landmarks) <= embedding_ids.max():
            continue
        points = landmarks[embedding_ids]
        if not np.isfinite(points).all() or not points.any():
            continue
        pose = np.asarray(frame["transform_matrix"])
        names.append(name)
        projections.append(intrinsic @ np.linalg.inv(pose)[:3])
        cameras.append(pose[:3, :3])
        observations.append(points)
    if len(names) < 2:
        raise ValueError("FLAME initialization requires at least two detected views.")
    chosen = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
        len(names), size=min(MAX_VIEWS, len(names)), replace=False
    ))
    print(f"MediaPipe detected {len(names)}/{len(frames)} views. "
          f"Selected {np.asarray(names)[chosen].tolist()}.", flush=True)
    return intrinsic, *(np.asarray(a)[chosen] for a in (projections, cameras, observations))


def reproject(projections, point):
    homogeneous = projections[:, :, :3] @ point + projections[:, :, 3]
    return homogeneous[:, :2] / homogeneous[:, 2:]


def triangulate(projections, observations):
    rows = (observations[:, :, None] * projections[:, 2:3] - projections[:, :2]).reshape(-1, 4)
    if np.linalg.matrix_rank(rows) < 3:
        raise ValueError("Degenerate camera geometry for landmark triangulation.")
    _, _, vectors = np.linalg.svd(rows, full_matrices=False)
    homogeneous = vectors[-1]
    if homogeneous[3] == 0:
        raise ValueError("Landmark triangulated to a point at infinity.")
    initial = homogeneous[:3] / homogeneous[3]
    fitted = least_squares(
        lambda point: (reproject(projections, point) - observations).ravel(),
        initial, loss="soft_l1", f_scale=ROBUST_SCALE_PX,
    ).x
    if not np.isfinite(fitted).all():
        raise ValueError("Non-finite triangulated landmark.")
    return fitted


def similarity(source, target):
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Degenerate FLAME similarity correspondences.")
    u, singular, vh = np.linalg.svd(b.T @ a)
    sign = np.array([1.0, 1.0, np.linalg.det(u @ vh)])
    rotation = (u * sign) @ vh
    scale = (singular @ sign) / np.square(a).sum()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("FLAME initialization requires a positive finite scale.")
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return rotation, scale, translation


def tensor(data):
    return torch.as_tensor(data, dtype=torch.float32, device="cuda")


def forward(flame, params, cameras):
    count = len(cameras)
    return flame(
        cameras=cameras,
        shape_params=params["shape"].expand(count, -1),
        expression_params=params["exp"].expand(count, -1),
        eye_pose_params=params["eyes"].expand(count, -1),
        jaw_pose_params=params["jaw"].expand(count, -1),
        eyelid_params=params["eyelids"].expand(count, -1),
        rot_params=params["R"].expand(count, -1),
        trans_params=params["t"].expand(count, -1),
        scale_params=params["scale"],
    )


@torch.no_grad()
def initialize(flame, config, embedding_ids, projections, observations):
    stable = np.flatnonzero(np.isin(embedding_ids, STABLE_IDS))
    columns, points = [], []
    for column in stable:
        present = np.any(observations[:, column] != 0, axis=-1)
        if present.sum() >= 2:
            columns.append(column)
            points.append(triangulate(projections[present], observations[present, column]))
    if len(columns) < 3:
        raise ValueError("FLAME initialization requires three triangulated landmarks.")

    identity = torch.eye(3, device="cuda")[None]
    rotation_6d = matrix_to_rotation_6d(identity)
    params = {
        "shape": tensor(np.zeros((1, config.num_shape_params))),
        "exp": tensor(np.zeros((1, config.num_exp_params))),
        "R": rotation_6d,
        "t": tensor([[0, 0, 0]]),
        "scale": tensor([1]),
        "eyes": rotation_6d.repeat(1, 2),
        "jaw": rotation_6d,
        "eyelids": tensor([[0, 0]]),
    }
    model = forward(flame, params, identity)[2][0, columns].cpu().numpy()
    rotation, scale, translation = similarity(model, np.asarray(points))
    params["R"] = matrix_to_rotation_6d(tensor(rotation)[None])
    params["scale"] = tensor([scale])

    # FLAME rotates about the root joint. Match its actual transformed points.
    rotated = forward(flame, params, identity)[2][0, columns].cpu().numpy()
    target = scale * (model @ rotation.T) + translation
    params["t"] = tensor((target - rotated).mean(axis=0, keepdims=True))
    print(f"Initialized FLAME from {len(columns)} 3D landmarks. Scale {scale:.5f}.", flush=True)
    return params


def fit(flame, params, cameras, projections, observations):
    rates = {"R": 0.004, "t": 0.005, "log_scale": 0.003, "shape": 0.018, "exp": 0.012}
    variables = {key: params[key].clone().requires_grad_() for key in rates if key != "log_scale"}
    variables["log_scale"] = params["scale"].log().requires_grad_()
    optimizer = torch.optim.Adam([{"params": [variables[key]], "lr": lr} for key, lr in rates.items()])
    iterations = 850
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    valid = observations.ne(0).any(dim=-1)
    for step in range(iterations):
        params.update({key: variables[key] for key in rates if key != "log_scale"})
        params["scale"] = variables["log_scale"].exp()
        landmarks = forward(flame, params, cameras)[2]
        homogeneous = (projections[:, None, :, :3] @ landmarks[..., None]).squeeze(-1)
        homogeneous = homogeneous + projections[:, None, :, 3]
        residual = homogeneous[..., :2] / homogeneous[..., 2:] - observations
        robust = (residual[valid].square().sum(-1) + ROBUST_SCALE_PX**2).sqrt() - ROBUST_SCALE_PX
        loss = robust.mean() + 2 * params["shape"].square().mean() + params["exp"].square().mean()
        if step % 100 == 0:
            print(f"FLAME step {step}/{iterations} loss {loss.item():.5f}", flush=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    params["scale"] = variables["log_scale"].exp()
    return {key: value.detach() for key, value in params.items()}


def main():
    config = parse_args()
    with np.load("flame/mediapipe/mediapipe_landmark_embedding.npz") as embedding:
        embedding_ids = embedding["landmark_indices"].astype(int)
    start = time.perf_counter()
    intrinsic, projections, cameras, observations = load_views(config, embedding_ids)
    print(f"Landmark detection took {time.perf_counter() - start:.2f}s.", flush=True)
    flame = FLAME(config).cuda()
    initial = initialize(flame, config, embedding_ids, projections, observations)
    cameras = tensor(cameras)
    params = fit(flame, initial, cameras, tensor(projections), tensor(observations))
    with torch.no_grad():
        params["act_trans"] = forward(flame, params, cameras)[3]
    params["intrinsic"] = tensor(intrinsic)
    params["texture"] = tensor(np.zeros((1, config.tex_params)))
    if not all(torch.isfinite(value).all().item() for value in params.values()):
        raise ValueError("FLAME fitting returned non-finite parameters.")
    destination = Path(config.save_root)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(params, destination / "fitting.pkl")
    print(f"Fitted FLAME scale {params['scale'].item():.5f}.", flush=True)


if __name__ == "__main__":
    main()
