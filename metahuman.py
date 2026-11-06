"""Export registered STFR geometry into the calibrated MetaHuman OBJ space.

Column-vector convention: p_target = scale * rotation @ p_stfr + translation.
The pipeline's source meshes/cameras stay in STFR space for reconstruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_TRANSFORM = Path(__file__).resolve().parent / 'misc/metahuman_transform.json'


def read_obj_vertices(path):
    vertices = []
    for line in Path(path).read_text().splitlines():
        fields = line.split('#', 1)[0].split()
        if fields and fields[0] == 'v':
            vertices.append(list(map(float, fields[1:4])))
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{path}: missing or invalid OBJ vertices')
    return result


def fit_similarity(source, target):
    """Least-squares positive uniform scale, proper rotation and translation."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('Expected corresponding Nx3 source and target vertices')
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Need at least three finite point correspondences')
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('Similarity fit requires non-collinear points')
    u, values, vt = np.linalg.svd(y.T @ x)
    correction = np.ones(3)
    correction[-1] = -1.0 if np.linalg.det(u @ vt) < 0 else 1.0
    rotation = (u * correction) @ vt
    scale = float(values @ correction / np.sum(x * x))
    if scale <= 0:
        raise ValueError('Fit did not produce a positive scale')
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return dict(scale=scale, rotation=rotation.tolist(), translation=translation.tolist())


def validate_transform(transform):
    scale = float(transform['scale'])
    rotation = np.asarray(transform['rotation'], dtype=float)
    translation = np.asarray(transform['translation'], dtype=float)
    if (not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3)
            or translation.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()):
        raise ValueError('Invalid similarity transform dimensions or values')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
        raise ValueError('Expected an orthonormal proper rotation (determinant +1)')
    return scale, rotation, translation


def transform_points(points, transform):
    scale, rotation, translation = validate_transform(transform)
    return scale * np.asarray(points) @ rotation.T + translation


def transform_obj(source, output, transform_path=DEFAULT_TRANSFORM):
    """Transform only v/vn records; retain UVs, faces, groups and materials."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('MetaHuman export must not overwrite its STFR-space source')
    transform = json.loads(Path(transform_path).read_text())
    scale, rotation, translation = validate_transform(transform)
    result, vertex_count, normal_count = [], 0, 0
    for line in source.read_text().splitlines(keepends=True):
        content, marker, comment = line.partition('#')
        fields = content.split()
        if not fields or fields[0] not in ('v', 'vn'):
            result.append(line)
            continue
        if len(fields) < 4:
            raise ValueError(f'{source}: incomplete {fields[0]} record')
        xyz = np.asarray(list(map(float, fields[1:4])))
        if not np.isfinite(xyz).all():
            raise ValueError(f'{source}: nonfinite {fields[0]} record')
        xyz = rotation @ xyz
        if fields[0] == 'v':
            xyz = scale * xyz + translation
            vertex_count += 1
        else:
            length = np.linalg.norm(xyz)
            if length <= 1e-15:
                raise ValueError(f'{source}: zero-length normal')
            xyz /= length
            normal_count += 1
        fields[1:4] = [format(float(value), '.17g') for value in xyz]
        rewritten = ' '.join(fields)
        if marker:
            rewritten += ' #' + comment.rstrip('\r\n')
        result.append(rewritten + '\n')
    if not vertex_count:
        raise ValueError(f'{source}: no vertices to export')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(result))
    return dict(output=str(output), vertices=vertex_count, normals=normal_count,
                transform=str(Path(transform_path).resolve()))


def export_final_mesh(data_root):
    """Export from the canonical source each time, so reruns never compound sRT."""
    dataset = Path(data_root)
    name = 'final_hack'
    source = dataset / (name + '.obj')
    if not source.is_file():
        raise FileNotFoundError(f'Cannot export MetaHuman geometry: {source} is missing')
    report = transform_obj(source, dataset / (name + '_metahuman.obj'))
    (dataset / (name + '_metahuman.transform.json')).write_text(
        json.dumps(dict(source=str(source), **report,
                        calibration=json.loads(DEFAULT_TRANSFORM.read_text())), indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, help='Registered sample_dataset directory')
    parser.add_argument('--input', type=Path, help='Alternatively transform a specific OBJ')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.data_root is not None and args.input is None and args.output is None:
        report = export_final_mesh(args.data_root)
    elif args.input is not None and args.output is not None and args.data_root is None:
        report = transform_obj(args.input, args.output)
    else:
        parser.error('Specify either --data_root or both --input and --output')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
