"""Append eye caps to registered ICT OBJ geometry without changing its UV atlas.

OBJ geometry and texture indices are deliberately independent: cap corners reuse
the eyelid vertices but receive new texture coordinates at the UV seam.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union


@dataclass
class ObjMesh:
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    face_uv: np.ndarray


def read_obj(path):
    vertices, uv, faces, face_uv = [], [], [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'vt':
            uv.append([float(x) for x in fields[1:3]])
        elif fields[0] == 'f':
            corners = [x.split('/') for x in fields[1:]]
            if any(len(x) < 2 or not x[1] for x in corners):
                raise ValueError(f'{path}: all faces must have UV coordinates')
            vi = [int(x[0]) - 1 if int(x[0]) > 0 else len(vertices) + int(x[0]) for x in corners]
            ti = [int(x[1]) - 1 if int(x[1]) > 0 else len(uv) + int(x[1]) for x in corners]
            for i in range(1, len(corners) - 1):
                faces.append([vi[0], vi[i], vi[i + 1]])
                face_uv.append([ti[0], ti[i], ti[i + 1]])
    mesh = ObjMesh(*(np.asarray(x, dtype=dtype) for x, dtype in
                     [(vertices, float), (uv, float), (faces, int), (face_uv, int)]))
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.uv).all():
        raise ValueError(f'{path}: empty or nonfinite geometry/UV')
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError(f'{path}: invalid geometry indices')
    if mesh.face_uv.min() < 0 or mesh.face_uv.max() >= len(mesh.uv):
        raise ValueError(f'{path}: invalid UV indices')
    return mesh


def write_obj(path, mesh, original_face_count):
    with Path(path).open('w', newline='\n') as f:
        f.write('# STFR registered ICT; eye caps have separate UV islands\n')
        for p in mesh.vertices:
            f.write('v ' + ' '.join(f'{x:.17g}' for x in p) + '\n')
        for p in mesh.uv:
            f.write('vt ' + ' '.join(f'{x:.17g}' for x in p) + '\n')
        f.write('s 1\ng original_ict\n')
        for i, (face, tex) in enumerate(zip(mesh.faces, mesh.face_uv)):
            if i == original_face_count:
                f.write('g eye_caps\n')
            f.write('f ' + ' '.join(f'{v + 1}/{t + 1}' for v, t in zip(face, tex)) + '\n')


def boundary_loops(faces):
    edges = [(int(a), int(b)) for f in faces for a, b in zip(f, np.roll(f, -1))]
    counts = Counter(tuple(sorted(e)) for e in edges)
    if max(counts.values()) > 2:
        raise ValueError('Mesh has nonmanifold edges')
    boundary = [(a, b) for a, b in edges if counts[tuple(sorted((a, b)))] == 1]
    next_vertex = dict(boundary)
    if len(next_vertex) != len(boundary) or len(set(next_vertex.values())) != len(boundary):
        raise ValueError('Mesh boundary is not a collection of oriented loops')
    loops = []
    while next_vertex:
        start = current = next(iter(next_vertex))
        loop = []
        while current in next_vertex:
            loop.append(current)
            current = next_vertex.pop(current)
        if current != start:
            raise ValueError('Open boundary chain')
        loops.append(np.asarray(loop))
    return loops


def ict_eye_loops(mesh, template_path=None):
    templates = ([Path(template_path)] if template_path else
                 sorted((Path(__file__).resolve().parents[1] / 'registration/assets').glob('*_eye_template.obj')))
    # Compare indexed topology before using template semantics. Never guess on a
    # remeshed input: Wrap's ICT vertex numbering is required.
    def topology(faces):
        return sorted(tuple(sorted(map(int, f))) for f in faces)
    key = topology(mesh.faces)
    for path in templates:
        template = read_obj(path)
        if len(mesh.vertices) != len(template.vertices) or key != topology(template.faces):
            continue
        loops = boundary_loops(template.faces)
        eyes = [loop for loop in loops if len(loop) == 42 and template.vertices[loop, 1].mean() > 0]
        if len(eyes) != 2:
            raise ValueError(f'{path}: expected exactly two ICT eye boundaries')
        actual = {frozenset(loop): loop for loop in boundary_loops(mesh.faces)}
        return [actual[frozenset(loop)] for loop in eyes], path.name
    raise ValueError('Input does not match an ICT template topology; pass --template for its matching ICT template')


def pack_islands(mesh, count, resolution=1024, padding_pixels=8):
    """Conservative raster search followed by exact polygon clearance checks."""
    if (mesh.uv < 0).any() or (mesh.uv > 1).any():
        raise ValueError('Original UV coordinates must lie in [0, 1]')
    triangles = [Polygon(mesh.uv[f]) for f in mesh.face_uv]
    occupied = unary_union([p for p in triangles if p.area > 1e-16] + [MultiPoint(mesh.uv)])
    image = Image.new('1', (resolution + 1, resolution + 1))
    draw = ImageDraw.Draw(image)
    for f in mesh.face_uv:
        draw.polygon([tuple(p) for p in mesh.uv[f] * resolution], fill=1)
    # Even unreferenced OBJ texture coordinates retain their reserved space.
    for p in mesh.uv * resolution:
        draw.point(tuple(p), fill=1)
    raster = np.asarray(image, dtype=np.uint8).copy()
    padding = padding_pixels / resolution
    boxes = []
    for _ in range(count):
        found = None
        for scale in np.linspace(1, .2, 33):
            width, height = np.ceil(np.array([.20, .12]) * scale * resolution).astype(int)
            # Include the requested gutter on all four sides of the candidate.
            w, h = width + 2 * padding_pixels, height + 2 * padding_pixels
            integral = np.pad(raster.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
            for y, x in np.argwhere(sums == 0)[::4]:
                bounds = np.array([x + padding_pixels, y + padding_pixels,
                                   x + padding_pixels + width, y + padding_pixels + height]) / resolution
                rectangle = box(*bounds)
                if rectangle.distance(occupied) >= padding and all(rectangle.distance(box(*b)) >= padding for b in boxes):
                    found = bounds
                    raster[y:y + h, x:x + w] = 1
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError('Could not fit eye UV islands with the requested padding; original UVs were not modified')
        boxes.append(found)
    return boxes, occupied, padding


def fill_eye_holes(input_path, output_path, template_path=None, rings=3, padding_pixels=8):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must differ from the original registered mesh')
    if rings < 1 or padding_pixels < 1:
        raise ValueError('Use at least one interior ring and one pixel of UV padding')
    mesh = read_obj(input_path)
    eyes, template = ict_eye_loops(mesh, template_path)
    boxes, occupied, padding = pack_islands(mesh, len(eyes), padding_pixels=padding_pixels)
    vertices, uv = list(mesh.vertices), list(mesh.uv)
    faces, face_uv = list(mesh.faces), list(mesh.face_uv)
    report = {'input': str(input_path), 'output': str(output_path), 'template': template,
              'uv_padding': padding, 'eyes': []}
    for loop, bounds in zip(eyes, boxes):
        # Reverse original boundary winding to give each shared edge two
        # oppositely oriented incident faces.
        loop = loop[::-1]
        points = mesh.vertices[loop]
        center = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        basis = axes[:2].copy()
        for axis in basis:
            if axis[np.argmax(np.abs(axis))] < 0:
                axis *= -1
        projected = (points - center) @ basis.T
        polygon = Polygon(projected)
        if not polygon.is_valid or polygon.area <= 1e-12:
            raise ValueError('Eye boundary has an invalid planar projection')
        # This ring construction requires a star-shaped polygon. Fail instead
        # of emitting folded caps if a pathological registration violates it.
        from shapely.geometry import LineString
        if not all(polygon.buffer(1e-10).covers(LineString([[0, 0], p])) for p in projected):
            raise ValueError('Eye boundary is not star shaped around its center')
        chart_min, chart_extent = projected.min(0), np.ptp(projected, axis=0)
        def chart(coords):
            return bounds[:2] + (coords - chart_min) / chart_extent * (bounds[2:] - bounds[:2])
        previous = loop.tolist()
        previous_uv = list(range(len(uv), len(uv) + len(loop)))
        uv.extend(chart(projected))
        face_start = len(faces)
        for ring in range(1, rings + 1):
            radius = 1 - ring / (rings + 1)
            current = list(range(len(vertices), len(vertices) + len(loop)))
            current_uv = list(range(len(uv), len(uv) + len(loop)))
            vertices.extend(center + radius * (points - center))
            uv.extend(chart(radius * projected))
            for j in range(len(loop)):
                k = (j + 1) % len(loop)
                faces.extend([[previous[j], previous[k], current[k]], [previous[j], current[k], current[j]]])
                face_uv.extend([[previous_uv[j], previous_uv[k], current_uv[k]], [previous_uv[j], current_uv[k], current_uv[j]]])
            previous, previous_uv = current, current_uv
        center_index, center_uv = len(vertices), len(uv)
        vertices.append(center)
        uv.append(chart(np.zeros(2)))
        for j in range(len(loop)):
            k = (j + 1) % len(loop)
            faces.append([previous[j], previous[k], center_index])
            face_uv.append([previous_uv[j], previous_uv[k], center_uv])
        report['eyes'].append({'boundary_vertices': loop.tolist(), 'uv_box': bounds.tolist(),
                               'first_face': face_start, 'face_count': len(faces) - face_start})
    result = ObjMesh(np.asarray(vertices), np.asarray(uv), np.asarray(faces), np.asarray(face_uv))
    caps = result.faces[len(mesh.faces):]
    area = np.linalg.norm(np.cross(result.vertices[caps[:, 1]] - result.vertices[caps[:, 0]],
                                   result.vertices[caps[:, 2]] - result.vertices[caps[:, 0]]), axis=1)
    if np.any(area <= 1e-14):
        raise ValueError('Eye cap contains a degenerate triangle')
    cap_uv = unary_union([Polygon(result.uv[f]) for f in result.face_uv[len(mesh.faces):]])
    overlap = cap_uv.intersection(occupied).area
    if overlap > 1e-14 or cap_uv.distance(occupied) < padding - 1e-10:
        raise ValueError('Eye UV islands overlap the original atlas or violate the gutter')
    if len(boundary_loops(result.faces)) != len(boundary_loops(mesh.faces)) - 2:
        raise ValueError('Eye cap topology validation failed')
    report.update(original_vertices=len(mesh.vertices), original_faces=len(mesh.faces),
                  original_uv_coordinates=len(mesh.uv), added_uv_coordinates=len(result.uv) - len(mesh.uv),
                  added_vertices=len(result.vertices) - len(mesh.vertices),
                  added_faces=len(result.faces) - len(mesh.faces), uv_overlap_area=overlap,
                  minimum_uv_clearance=cap_uv.distance(occupied))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_obj(output_path, result, len(mesh.faces))
    output_path.with_suffix('.eye_caps.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--rings', type=int, default=3)
    parser.add_argument('--padding_pixels', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(fill_eye_holes(args.input, args.output, args.template, args.rings, args.padding_pixels), indent=2))
