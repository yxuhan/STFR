"""Continuous texture queries with an explicit world-to-training-coordinate map."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

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


def read_appearance(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    matrix = np.asarray(data['world_to_texture'], dtype=float)
    if data.get('version') != 1 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'Invalid volume appearance descriptor: {path}')
    if not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-15:
        raise ValueError('Invalid world-to-texture affine map')
    return data


def appearance_mesh(path):
    return (Path(path).resolve().parent / read_appearance(path)['mesh']).resolve()


def transform_appearance(source, output, mesh_path, new_from_old):
    """Geometry changes frames; the continuous color field stays in training space."""
    source, output = Path(source).resolve(), Path(output).resolve()
    data = read_appearance(source)
    data['world_to_texture'] = (np.asarray(data['world_to_texture']) @ np.linalg.inv(new_from_old)).tolist()
    data['checkpoint'] = os.path.relpath((source.parent / data['checkpoint']).resolve(), output.parent)
    data['mesh'] = os.path.relpath(Path(mesh_path).resolve(), output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n')


def query_xyz_map(xyz, mask, network, world_to_texture, chunk_size=65536):
    """Query visible pixels only; interpolate positions BEFORE nonlinear sampling."""
    if xyz.ndim != 4 or xyz.shape[1] != 3 or mask.shape != (xyz.shape[0], 1, *xyz.shape[2:]):
        raise ValueError('Expected xyz Bx3xHxW and mask Bx1xHxW')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    points = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    valid = mask.reshape(-1).bool()
    colors = torch.zeros_like(points)
    matrix = torch.as_tensor(world_to_texture, dtype=points.dtype, device=points.device)
    indices = torch.where(valid)[0]
    with torch.no_grad():
        for indices_chunk in indices.split(chunk_size):
            if indices_chunk.numel():
                positions = points[indices_chunk] @ matrix[:3, :3].T + matrix[:3, 3]
                colors[indices_chunk] = network(positions).to(colors).clamp(0, 1)
    return colors.reshape(xyz.shape[0], xyz.shape[2], xyz.shape[3], 3).permute(0, 3, 1, 2).contiguous()


class VolumeTexture(nn.Module):
    '''
    map a 3d position to its BRDF parameters
    '''
    def __init__(
        self,
        num_levels=16,
        level_dim=4,
    ):
        super().__init__()
        import tinycudann as tcnn
        per_level_scale = np.exp2(np.log2(1024 / 16) / (num_levels - 1))
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',
                'n_levels': num_levels,
                'n_features_per_level': level_dim,
                'log2_hashmap_size': 14,
                'base_resolution': 16,
                'per_level_scale': per_level_scale,
            }
        )

        n_output_dims = 3
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims + 3,
            n_output_dims=n_output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def forward(self, x):
        h = self.encoding(x).to(x)
        h = torch.cat([2 * x - 1, h], dim=-1)
        out = self.network(h).to(h)
        return out.abs()


class VolumeTextureAppearance:
    def __init__(self, path, device='cuda:0'):
        path = Path(path).resolve()
        data = read_appearance(path)
        self.network = VolumeTexture(**data['model']).to(device).eval()
        self.network.load_state_dict(torch.load(path.parent / data['checkpoint'], map_location=device))
        self.matrix = torch.as_tensor(data['world_to_texture'], dtype=torch.float32, device=device)

    def render(self, xyz, mask, chunk_size=65536):
        return query_xyz_map(xyz, mask, self.network, self.matrix, chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_path', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--normalization')
    args = parser.parse_args()
    prepare_appearance(args.mesh_path, args.ckpt_path, args.output, args.normalization)
