'''
为fitting提供初始化head pose and scale参数
'''


import sys
sys.path.append(".")
sys.path.append("..")

import cv2
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import json
import trimesh
from scipy.spatial.transform import Rotation

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from utils.mesh_renderer import MeshRenderer


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str)
parser.add_argument('--mesh_path', type=str)
parser.add_argument('--cam_path', type=str)
parser.add_argument('--save_root', type=str)
parser.add_argument('--appearance_path', type=str, required=True)
opt = parser.parse_args()

opt, _ = parser.parse_known_args()


img_root = opt.img_root
os.makedirs(opt.save_root, exist_ok=True)


def draw_all_lmk(img, lmks):
    img_copy = img.copy()
    for lmk in lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy


class LandmarksDetectorIBug:
	def __init__(self, device):
		self.face_detector = RetinaFacePredictor(
			threshold=0.8, device='cuda:0',
			model=RetinaFacePredictor.get_model('resnet50')
		)

		# Create a facial landmark detector
		self.landmark_detector = FANPredictor(
			device=device, model=FANPredictor.get_model('2dfan2_alt')
		)
	
	def detect(self, images):
		# images should be in OPENCV format
		detected_faces = self.face_detector(images, rgb=False)
		landmarks, scores = self.landmark_detector(images, detected_faces, rgb=False)
		return landmarks


def compute_landmark_width(landmarks):
	left_eye_right = landmarks[0][39]
	right_eye_left = landmarks[0][42]

	left_eye_left = landmarks[0][36]
	right_eye_right = landmarks[0][45]

	left_mouth = landmarks[0][48]
	right_mouth = landmarks[0][54]

	# metric = right_eye_right[1] - left_eye_left[1] + \
	# 	right_mouth[0] - left_mouth[0] + \
	# 	right_eye_left[0] - left_eye_right[0]

	right_eye_width = right_eye_right[0] - right_eye_left[0]
	left_eye_width = left_eye_right[0] - left_eye_left[0]
	metric = min(right_eye_width, left_eye_width) / max(right_eye_width, left_eye_width)

	vis_ldm = np.stack([
		left_eye_left, left_eye_right,
		right_eye_left, right_eye_right,
		left_mouth, right_mouth,
	], axis=0)
	return vis_ldm, metric


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


def align_verts(canonical_verts, target_verts):
	'''
	all numpy array with the same shape [v,3]
	'''        
	# move to the center
	canonical_verts_mean = np.mean(canonical_verts, axis=0, keepdims=True)
	target_verts_mean = np.mean(target_verts, axis=0, keepdims=True)
	canonical_verts_centered = canonical_verts - canonical_verts_mean
	target_verts_centered = target_verts - target_verts_mean

	# compute scale
	canonical_scale = np.sqrt(
		np.mean(
			np.sum(canonical_verts_centered ** 2, axis=-1), 
			axis=0
		)
	)
	target_scale = np.sqrt(
		np.mean(
			np.sum(target_verts_centered ** 2, axis=-1), 
			axis=0
		)
	)
	relative_scale = canonical_scale / target_scale
	target_verts_centered_scaled = target_verts_centered * relative_scale

	# compute rotation
	rot, _ = Rotation.align_vectors(canonical_verts_centered, target_verts_centered_scaled)
	rot = rot.as_matrix()	
	trans = canonical_verts_mean - (rot @ (target_verts_mean * relative_scale)[..., None])[..., 0]
	return relative_scale, rot, trans


# select frontal view
landmark_detector = LandmarksDetectorIBug(device='cuda:0')
best_info = {
	"img_path": None,
	"metric": -1,
	"landmarks": None,
}
for img_pth in tqdm(os.listdir(img_root)):
	try:
		img = cv2.imread(os.path.join(img_root, img_pth))
		landmarks = landmark_detector.detect(img)

		vis_ldm, metric = compute_landmark_width(landmarks)
		# img_vis = draw_all_lmk(img, vis_ldm)
		# cv2.imwrite("%05d.jpg" % i, img_vis)

		if metric > best_info["metric"]:
			best_info["metric"] = metric
			best_info["img_path"] = img_pth
			best_info["landmarks"] = landmarks
	except:
		print("landmark detect error in %s" % img_pth)
		continue


select_ldm_index_list = [
	36, 39,  # left eye
	42, 45,  # right eye
	48, 54,  # mouth
	31, 35,  # nose
]

best_img_path = os.path.join(img_root, best_info["img_path"])
front_img = cv2.imread(best_img_path)
front_img = draw_all_lmk(front_img, best_info["landmarks"][0][select_ldm_index_list])
cv2.imwrite(
	os.path.join(opt.save_root, "frontal_view.jpg"),
	front_img,
)


meta_file_path = os.path.join(opt.cam_path)
with open(meta_file_path, 'r') as f:
	meta = json.load(f)

HEIGHT = int(meta['h'])
WIDTH = int(meta['w'])

# load intrinsics
intrinsic = np.eye(3, dtype=np.float32)
intrinsic[0, 0] = meta['fl_x']
intrinsic[1, 1] = meta['fl_y']
intrinsic[0, 2] = meta['cx']
intrinsic[1, 2] = meta['cy']
intrinsic = torch.from_numpy(intrinsic).cuda()
intrinsic_rel = torch.clone(intrinsic)
intrinsic_rel[0] /= WIDTH
intrinsic_rel[1] /= HEIGHT

# load extrinsics
frames = meta["frames"]
for f_id in tqdm(range(len(frames))):
	cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
	# cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.0)
	img_name = os.path.basename(frames[f_id]['file_path'])
	if img_name == best_info["img_path"]:
		cam_pose = torch.from_numpy(cur_pose).cuda()
		break

# render point map
device = torch.device("cuda:0")
mesh = trimesh.load_mesh(opt.mesh_path)
vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
vertices = vertices[None, ...]  # [1,v,3]
faces = faces[None, ...]  # [1,v,3]

mesh_renderer = MeshRenderer(device=device)

mesh_dict = {
	"faces": faces,
	"vertice": vertices,
	"attributes": vertices,
	"size": (HEIGHT, WIDTH),
}

res, _ = mesh_renderer.render_mesh(
	mesh_dict, intrinsic_rel[None, ...], torch.inverse(cam_pose)[None, :3]
)

# compuite 3D correspondences
landmarks = best_info["landmarks"]
ldm_3d_detailed_mesh = []
for ldm_idx in select_ldm_index_list:
	ldm_2d = landmarks[0][ldm_idx]
	ldm_3d = res[
		0, :, int(ldm_2d[1]), int(ldm_2d[0])
	]
	ldm_3d_detailed_mesh.append(ldm_3d)
ldm_3d_detailed_mesh = torch.stack(ldm_3d_detailed_mesh, dim=0).cpu().numpy()  # [n,2]
ldm_3d_template = torch.load("align/canonical_lmk68.pkl", map_location="cpu")[select_ldm_index_list].detach().numpy()

s, R, t = align_verts(ldm_3d_detailed_mesh, ldm_3d_template)

torch.save(
    {"rot": R, "trans": t}, 
    os.path.join(opt.save_root, "align_init.pkl")\
)

re_scale = 1 / s
from registration.volume_texture import transform_appearance
scaled_path = os.path.join(opt.save_root, 'align_scaled.ply')
trimesh.Trimesh(vertices=mesh.vertices * re_scale, faces=mesh.faces, process=False).export(scaled_path)
frame_change = np.eye(4)
frame_change[:3, :3] *= re_scale
transform_appearance(opt.appearance_path, os.path.join(opt.save_root, 'appearance.json'), scaled_path, frame_change)

frames = meta["frames"]
for f_id in tqdm(range(len(frames))):
	cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
	cur_pose[:3, 3] *= re_scale
	frames[f_id]['transform_matrix'] = cur_pose.tolist()

meta['frames'] = frames
with open(os.path.join(opt.save_root, "align_scaled.json"), 'w') as f:
	json.dump(meta, f, indent=4)
