import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import math
import trimesh
import torch
from torchvision.utils import save_image
import json
from pytorch3d.io import load_objs_as_meshes
import argparse
import cv2
from scipy.spatial.transform import Rotation

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from utils.mesh_renderer import MeshRenderer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="xxx")
parser.add_argument('--save_root', type=str, default="xxx")
parser.add_argument('--ldm_type', type=str, default="eyelid", choices=["ibug", "eyelid", "interp"])
parser.add_argument('--reg_template', choices=['close', 'open', 'fill'], default='open')
parser.add_argument('--appearance_path', type=str, required=True)
opt, _ = parser.parse_known_args()


def load_ict_model(device = torch.device('cpu'), template='open'):
	# origin ICT, see https://github.com/ICT-VGL/ICT-FaceKit
	# lm_index = np.array([
	#     1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966,
	# ])
	# mesh_pth = "ICT/ict_origin.obj"  # origin ICT

	# ICT remove cavaity
	if template == 'close':
		lm_index = np.array([
			# contour
			1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 
			# eyebrow
			268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 
			# nose
			978, 4527, 4766, 4733, 1140, 2075, 1147, 4269, 3360, 
			# left eye
			1507, 1542, 1537, 1528, 1518, 1511, 
			# right eye
			3742, 3751, 3756, 3721, 3725, 3732, 
			# mouth outer
			5276, 5263, 2081, 0, 4275, 5768, 5781, 5914, 6029, 5086, 5525, 5409, 
			# mouth inner
			5270, 5279, 5101, 5784, 5775, 6038, 5085, 5534, 
		])
		mesh_pth = "assets/close_eye_template.obj"  # ICT remove cavaity
	else:
		lm_index = np.array([
			# contour 17
			2053, 1593, 1727, 729, 940, 839, 1421, 824, 1492, 3068, 2926, 3085, 3186, 2970, 3967, 3822, 4277, 
			# eyebrow 10
			460, 835, 534, 2170, 2146, 4365, 4384, 2768, 2754, 2697, 
			# nose 9
			1553, 4537, 4798, 4782, 97, 2217, 1673, 4424, 2347, 
			# left eye 6
			1573, 1288, 1201, 1141, 1068, 1469, 
			# right eye 6
			4163, 3376, 3541, 3802, 3306, 3322, 
			# mouth outer 12
			5182, 5229, 2222, 5399, 4429, 5758, 5743, 5887, 5964, 5481, 5448, 5365, 
			# mouth inner 8
			5179, 5250, 5325, 5778, 5732, 5943, 5480, 5426, 
		])
		mesh_pth = "assets/open_eye_template.obj"

	if template == 'fill':
		lm_index = np.asarray(json.loads(Path('assets/fill_eye_landmarks.json').read_text())['ibug'])
		mesh_pth = 'assets/fill_eye_template.obj'
	lm_index = torch.from_numpy(lm_index).long().to(device)
	ict_mesh = load_objs_as_meshes([mesh_pth], device=device)
	# norm_mesh, _ = normalize_mesh(ict_mesh)
	return ict_mesh, lm_index[None, ...]


def load_ict_model_eyelid(device = torch.device('cpu'), template='open'):
	# origin ICT, see https://github.com/ICT-VGL/ICT-FaceKit
	# lm_index = np.array([
	#     1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966,
	# ])
	# mesh_pth = "ICT/ict_origin.obj"  # origin ICT

	# ICT remove cavaity
	if template == 'close':
		lm_index = np.array([
			# contour 17
			1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 
			# eyebrow 10
			268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 
			# nose 9
			978, 4527, 4766, 4733, 1140, 2075, 1147, 4269, 3360, 
			# mouth outer 12
			5276, 5263, 2081, 0, 4275, 5768, 5781, 5914, 6029, 5086, 5525, 5409, 
			# mouth inner 8
			5270, 5279, 5101, 5784, 5775, 6038, 5085, 5534, 
			# left down
			1507, 1623, 1509, 1511, 1513, 1515, 1517, 1519, 1521, 1523, 1525, 1528, 
			# left up
			1507, 1545, 1579, 1577, 1575, 1539, 1931, 1536, 1534, 1532, 1530, 1528, 
			# right down
			3742, 3739, 3737, 3735, 3733, 3731, 3729, 3727, 3725, 3723, 3832, 3721, 
			# right up
			3742, 3744, 3746, 3748, 3750, 4131, 3753, 3789, 3791, 3793, 3759, 3721, 
		])
		mesh_pth = "assets/close_eye_template.obj"  # ICT remove cavaity
	else:
		lm_index = np.array([
			# contour 17
			2053, 1593, 1727, 729, 940, 839, 1421, 824, 1492, 3068, 2926, 3085, 3186, 2970, 3967, 3822, 4277, 
			# eyebrow 10
			460, 835, 534, 2170, 2146, 4365, 4384, 2768, 2754, 2697, 
			# nose 9
			1553, 4537, 4798, 4782, 97, 2217, 1673, 4424, 2347, 
			# # left eye 6
			# 1573, 1288, 1201, 1141, 1068, 1469, 
			# # right eye 6
			# 4163, 3376, 3541, 3802, 3306, 3322, 
			# mouth outer 12
			5182, 5229, 2222, 5399, 4429, 5758, 5743, 5887, 5964, 5481, 5448, 5365, 
			# mouth inner 8
			5179, 5250, 5325, 5778, 5732, 5943, 5480, 5426, 
			# left down
			1573, 1290, 1063, 1469, 1186, 1067, 1070, 1197, 1078, 1875, 1087, 1141, 
			# left up
			1573, 1113, 1262, 1574, 1246, 2123, 1200, 1117, 1129, 1145, 1142, 1141, 
			# right down
			4163, 3340, 4108, 3330, 3452, 3323, 4159, 3309, 3306, 3316, 3542, 3802, 
			# right up
			4163, 3396, 3400, 3384, 3374, 3455, 4341, 3501, 3804, 3515, 3369, 3802, 
		])
		mesh_pth = "assets/open_eye_template.obj"

	if template == 'fill':
		lm_index = np.asarray(json.loads(Path('assets/fill_eye_landmarks.json').read_text())['eyelid'])
		mesh_pth = 'assets/fill_eye_template.obj'
	lm_index = torch.from_numpy(lm_index).long().to(device)
	ict_mesh = load_objs_as_meshes([mesh_pth], device=device)
	# norm_mesh, _ = normalize_mesh(ict_mesh)
	return ict_mesh, lm_index[None, ...]


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


class LandmarksDetectorEyelid:
	def __init__(self, device):
		from eyelid_detector.eyelid_detector import LandmarkDetectorV3
		self.lmk_detector = LandmarkDetectorV3(
			"pretrained/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx", 
			use_onnx=True, 
			use_filter=False,
		)
		self.face_lmk_detector = LandmarksDetectorIBug(device=device)

	def detect(self, images):
		face_lmk = self.face_lmk_detector.detect(images)
		  
		kps = np.concatenate([
			np.mean(face_lmk[0, 36:42], axis=0, keepdims=True),  # left eye
			np.mean(face_lmk[0, 42:48], axis=0, keepdims=True),  # right eye
			face_lmk[0, 30:31],
			face_lmk[0, 57:58],
		], axis=0)
		
		eyelid_lmks, interval_det = self.lmk_detector(images[..., [2, 1, 0]], kps)
		eyelid_lmks = eyelid_lmks[68+38:]

		return self.make_landmarks(face_lmk, eyelid_lmks)

	def make_landmarks(self, face_lmk, eyelid_lmks):
		'''
		face_lmk: [1, 68, 2]
		eyelid_lmks: [1, 48+38, 2]
		return: [1, 68+48+38, 2]
		'''
		face_lmk = face_lmk[0]
		face_lmk_1 = face_lmk[:36]
		face_lmk_2 = face_lmk[48:]
		face_lmk_wo_eyes = np.concatenate([face_lmk_1, face_lmk_2], axis=0)  # [56,2]

		left_down = eyelid_lmks[:12]
		left_up = eyelid_lmks[12:24]
		right_down = eyelid_lmks[24:36][::-1]
		right_up = eyelid_lmks[36:48][::-1]

		eyelid_new = np.concatenate([left_down, left_up, right_down, right_up], axis=0)  # [48,2]

		landmark = np.concatenate([face_lmk_wo_eyes, eyelid_new], axis=0)
		return landmark[None, ...]


class LandmarksDetectorIBugInterp:
	def __init__(self, device):
		self.face_detector = RetinaFacePredictor(
			threshold=0.8, device='cuda:0',
			model=RetinaFacePredictor.get_model('resnet50')
		)

		# Create a facial landmark detector
		self.landmark_detector = FANPredictor(
			device=device, model=FANPredictor.get_model('2dfan2_alt')
		)
	
	def fit_cubic_and_sample(self, points):
		import matplotlib.pyplot as plt

		x = np.array([p[0] for p in points], dtype=float)
		y = np.array([p[1] for p in points], dtype=float)
				
		coefficients = np.polyfit(x, y, 3)
		cubic_func = np.poly1d(coefficients)
		
		min_x = np.min(x)
		max_x = np.max(x)
		
		sample_x = np.linspace(min_x, max_x, 12)
		
		sample_y = cubic_func(sample_x)
		
		sample_points = list(zip(sample_x, sample_y))
				
		return np.array(sample_points)
	
	def make_single_eyelid_line(self, ldm_4):
		return self.fit_cubic_and_sample(ldm_4)
	
	def make_eyelid(self, landmarks):
		face_lmk = landmarks[0]
		face_lmk_1 = face_lmk[:36]
		face_lmk_2 = face_lmk[48:]
		face_lmk_wo_eyes = np.concatenate([face_lmk_1, face_lmk_2], axis=0)  # [56,2]

		left_eye_up_line = face_lmk[36:40]
		left_eye_down_line = np.stack([
			face_lmk[36], face_lmk[41], face_lmk[40], face_lmk[39],
		], axis=0)

		right_eye_up_line = face_lmk[42:46]
		right_eye_down_line = np.stack([
			face_lmk[42], face_lmk[47], face_lmk[46], face_lmk[45],
		], axis=0)

		left_eye_down_ldm = self.make_single_eyelid_line(left_eye_down_line)
		left_eye_up_ldm = self.make_single_eyelid_line(left_eye_up_line)
		right_eye_down_ldm = self.make_single_eyelid_line(right_eye_down_line)
		right_eye_up_ldm = self.make_single_eyelid_line(right_eye_up_line)

		all_ldm = np.concatenate([
			face_lmk_wo_eyes,
			left_eye_down_ldm, left_eye_up_ldm,
			right_eye_down_ldm, right_eye_up_ldm,
		], axis=0)
		return all_ldm[None, ...]
	
	def detect(self, images):
		# images should be in OPENCV format
		detected_faces = self.face_detector(images, rgb=False)
		landmarks, scores = self.landmark_detector(images, detected_faces, rgb=False)
		landmarks = self.make_eyelid(landmarks)
		return landmarks


def fov_to_cam_int(fov_deg):
	image_width = 1. # 图像宽度
	image_height = 1.  # 图像高度

	# 计算主点坐标（图像中心）
	cx = image_width / 2
	cy = image_height / 2

	# 将视场角转换为弧度
	fov_rad = math.radians(fov_deg)

	# 计算焦距（以像素为单位）
	# 对于正方形图像和对称FOV，fx和fy相等
	fx = (image_width / 2) / math.tan(fov_rad / 2)
	fy = (image_height / 2) / math.tan(fov_rad / 2)

	# 构建内参矩阵
	intrinsic_matrix = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	])
	return intrinsic_matrix


def draw_all_lmk(img, lmks):
	img_copy = img.copy()
	for lmk in lmks:
		x = int(lmk[0] + 0.5)
		y = int(lmk[1] + 0.5)
		cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
	return img_copy


os.makedirs(opt.save_root, exist_ok=True)
wrap_root = opt.save_root

landmark_type = opt.ldm_type
if landmark_type in ["eyelid", "interp"]:
	if landmark_type == "interp":
		wrap_landmark_detector = LandmarksDetectorIBugInterp(device='cuda:0')
	else:
		wrap_landmark_detector = LandmarksDetectorEyelid(device='cuda:0')
	load_template_func = load_ict_model_eyelid
	opt.dense_eye = 1
	# mouth_inner = [17 + 10 + 9 + 12, 17 + 10 + 9 + 12 + 8]
	mouth_inner_min = 17 + 10 + 9 + 12
	mouth_inner_max = 17 + 10 + 9 + 12 + 8
else:
	wrap_landmark_detector = LandmarksDetectorIBug(device='cuda:0')
	load_template_func = load_ict_model
	opt.dense_eye = 0
	# mouth_inner = [17 + 10 + 9 + 12 + 12, 17 + 10 + 9 + 12 + 12 + 8]
	mouth_inner_min = 17 + 10 + 9 + 12 + 12
	mouth_inner_max = 17 + 10 + 9 + 12 + 12 + 8


#########################
# render point map
#########################
device = torch.device("cuda:0")
from registration.volume_texture import VolumeTextureAppearance, appearance_mesh
geometry_path = str(appearance_mesh(opt.appearance_path))
appearance = VolumeTextureAppearance(opt.appearance_path, device)
mesh = trimesh.load_mesh(geometry_path)
# Wrap's LoadGeom accepts OBJ, but not PLY. Convert only at this interface;
# no vertex colors are baked or carried into this geometry-only adapter.
wrap_geometry_path = os.path.join(wrap_root, 'target_geometry.obj')
trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).export(wrap_geometry_path)
vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
vertices = vertices[None, ...]  # [1,v,3]
faces = faces[None, ...]  # [1,v,3]

mesh_renderer = MeshRenderer(device=device)

mesh_dict = {
	"faces": faces,
	"vertice": vertices,
	"attributes": torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1),
	"size": (512, 512),
}

intrinsic_rel = fov_to_cam_int(20)
intrinsic_rel = torch.from_numpy(intrinsic_rel).float().to(device)
cam_rot = Rotation.from_euler('xyz', [-180, 0, 0], degrees=True).as_matrix()
cam_trans = np.array([0, 0, 5])
cam_pose = np.eye(4)
cam_pose[:3, :3] = cam_rot
cam_pose[:3, 3] = cam_trans
cam_pose = torch.from_numpy(cam_pose).float().to(device)

res, pix_to_face = mesh_renderer.render_mesh(
	mesh_dict, intrinsic_rel[None, ...], torch.inverse(cam_pose)[None, :3]
)
res_save_path = os.path.join(wrap_root, "rendered.png")
visible = (pix_to_face[..., 0] >= 0).unsqueeze(1)
rgb = appearance.render(res[:, :3], visible)
torch.save(res[:, :3].cpu(), os.path.join(wrap_root, 'frontal_xyz.pt'))
save_image(visible.float(), os.path.join(wrap_root, 'frontal_mask.png'))
save_image(rgb, os.path.join(wrap_root, 'frontal_rgb.png'))
save_image(rgb, res_save_path)
res_cv2 = cv2.imread(res_save_path)

eyelid_landmark = wrap_landmark_detector.detect(res_cv2)
landmarks = eyelid_landmark[0]

res_cv2 = draw_all_lmk(res_cv2, landmarks)
cv2.imwrite(res_save_path, res_cv2)

tgt_data = []
for i in range(len(landmarks)):
	if i < 17: continue
	if i >= mouth_inner_min and i < mouth_inner_max: continue  # mouth inner
	cur_verts = res[
		0, :3, int(landmarks[i][1]), int(landmarks[i][0])
	]
	tgt_data.append(
		{"x": cur_verts[0].item(), "y": cur_verts[1].item(), "z": cur_verts[2].item()}
	)

with open(os.path.join(wrap_root, "tgt.json"), "w") as f:
	json.dump(tgt_data, f)


##############
# proces template
ict_mesh, bfm_lm_index = load_template_func(
	torch.device('cuda:0'), template=opt.reg_template
)
bfm_data = []
mesh_pth = f"assets/{opt.reg_template}_eye_template.obj"
for i in range(len(bfm_lm_index[0])):
	if i < 17: continue  # contour
	if i >= mouth_inner_min and i < mouth_inner_max: continue  # mouth inner
	cur_verts = ict_mesh.verts_padded()[0, bfm_lm_index[0][i]]
	bfm_data.append(
		{"x": cur_verts[0].item(), "y": cur_verts[1].item(), "z": cur_verts[2].item()}
	)

with open(os.path.join(wrap_root, "bfm.json"), "w") as f:
	json.dump(bfm_data, f)

import json
with open("assets/template_all_wrap.wrap", "r") as f:
	wrap_cfg = json.load(f)

wrap_cfg["nodes"]["LoadGeom"]["params"]["fileName"]["value"] = os.path.abspath(mesh_pth)
wrap_cfg["nodes"]["LoadGeom1"]["params"]["fileName"]["value"] = os.path.abspath(wrap_geometry_path)
wrap_cfg["nodes"]["LoadPoints"]["params"]["fileName"]["value"] = os.path.join(wrap_root, "bfm.json")
wrap_cfg["nodes"]["LoadPoints1"]["params"]["fileName"]["value"] = os.path.join(wrap_root, "tgt.json")
wrap_cfg["nodes"]["SaveGeom"]["params"]["fileName"]["value"] = os.path.join(wrap_root, "final_hack.obj")

with open(os.path.join(wrap_root, "wrap.wrap"), "w") as f:
	json.dump(wrap_cfg, f, indent=4)
