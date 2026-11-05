"""Continuous texture queries with an explicit world-to-training-coordinate map."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_appearance(mesh_path, checkpoint, output, normalization=None):
    """Use training's recorded normalization; also support existing checkpoints."""
    import trimesh
    mesh_path, checkpoint, output = map(lambda p: Path(p).resolve(), (mesh_path, checkpoint, output))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    mesh_hash = file_sha256(mesh_path)
    if normalization and Path(normalization).is_file():
        info = json.loads(Path(normalization).read_text())
        if info['mesh_sha256'] != mesh_hash:
            raise ValueError('Volume normalization belongs to a different source mesh')
        offset, scale = np.array(info['offset']), float(info['scale'])
    else:
        # Match render_position_map's original loader and float32 GPU reductions.
        mesh = trimesh.load_mesh(mesh_path)
        vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device='cuda')
        offset = vertices.mean(0).cpu().numpy().astype(float)
        scale = float((2 / (vertices.max(0).values - vertices.min(0).values).max()).item())
    matrix = np.eye(4)
    matrix[:3, :3] *= scale / 2
    matrix[:3, 3] = .5 - offset * scale / 2
    data = dict(version=1, type='stfr_volume_texture', mesh=os.path.relpath(mesh_path, output.parent),
                checkpoint=os.path.relpath(checkpoint, output.parent), model=dict(num_levels=16, level_dim=4),
                world_to_texture=matrix.tolist(), source_mesh_sha256=mesh_hash,
                training_normalization=dict(offset=offset.tolist(), scale=scale))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Continuous volume appearance: {output}', flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_path', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--normalization')
    args = parser.parse_args()
    prepare_appearance(args.mesh_path, args.ckpt_path, args.output, args.normalization)
