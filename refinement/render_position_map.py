'''
render uv img for real dataset
'''


import sys
sys.path.append(".")
sys.path.append("..")

import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
import json
import yaml
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import kornia
from scipy.spatial.transform import Rotation

from mesh_renderer import MeshRenderer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--syn', type=int, default=1)
parser.add_argument('--cam_path', type=str, default="xxx")
parser.add_argument('--mesh_path', type=str, default="xxx")
parser.add_argument('--img_root', type=str, default="xxx")
parser.add_argument('--save_root', type=str, default="xxx")
opt, _ = parser.parse_known_args()


opt.device = "cuda"


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class MetaShapeDataset(Dataset):
    def __init__(self, device, meta_file_path, mode="train", cache_image=True):
        super().__init__()
        self.id_root = os.path.dirname(meta_file_path)
        self.device = device
        self.mode = mode
        self.cache_image = cache_image

        with open(meta_file_path, 'r') as f:
            meta = json.load(f)

        self.HEIGHT = int(meta['h'])
        self.WIDTH = int(meta['w'])

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = meta['fl_x']
        self.intrinsic[1, 1] = meta['fl_y']
        self.intrinsic[0, 2] = meta['cx']
        self.intrinsic[1, 2] = meta['cy']
        self.intrinsic = torch.from_numpy(self.intrinsic).to(self.device)
        self.intrinsic_rel = torch.clone(self.intrinsic)
        self.intrinsic_rel[0] /= self.WIDTH
        self.intrinsic_rel[1] /= self.HEIGHT
        
        # split dataset
        frames = meta["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])
        if mode in ["val"]:
            frames = frames[::10]
        self.frames = frames
        self.num_frames = len(self.frames)

        image_dir = opt.img_root

        # load per-frame meta information
        self.cam2world = []
        self.images, self.masks, self.spec_images = [], [], []
        self.img_name_list = []
        for f_id in tqdm(range(self.num_frames)):
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            if opt.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            img_name = os.path.basename(frames[f_id]['file_path'])
            self.img_name_list.append(img_name)
            if self.cache_image:
                img = self._load_img(os.path.join(image_dir, img_name))
                self.images.append(img)
            else:
                # self.spec_images.append(os.path.join(spec_image_dir, img_name))
                self.images.append(os.path.join(image_dir, img_name))

        if self.cache_image:
            self.images = torch.stack(self.images, dim=0)  # [b,3,h,w]
        
        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)  # [nf,4,4]
        self.cam2world = torch.from_numpy(self.cam2world).to(self.device)  # [nf,4,4]

        print("Create [%s] dataset, total [%d] frames" % (self.mode, self.num_frames))

    def __len__(self):
        return self.num_frames
        
    def _load_img(self, pth):
        if os.path.exists(pth):
            img = transforms.ToTensor()(Image.open(pth))[:3]
            return img.to(self.device)
        else:
            return torch.zeros(3, self.HEIGHT, self.WIDTH).to(self.device)
        
    def __getitem__(self, index):
        if self.cache_image:
            img = self.images[index]
        else:
            img = self._load_img(self.images[index])

        c2w = self.cam2world[index]
        
        # origins, viewdirs = self._compute_ray(c2w)
        info = {
            "pixels": img,  # [3,h,w]
            "c2w": c2w,  # [4,4]
            "intrinsic": self.intrinsic,
            "intrinsic_rel": self.intrinsic_rel,
            "img_name": self.img_name_list[index],
        }
        return info


class DiffusionSampler:
    def __init__(self, opt):
        self.device = opt.device
        self.mesh_renderer = MeshRenderer(self.device)

        # set dataset
        train_data = MetaShapeDataset(
            device=self.device, meta_file_path=opt.cam_path, mode="train",
        )
        self.dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
        data_root = opt.save_root
        self._load_geometry(opt.mesh_path)

        self.HEIGHT = train_data.HEIGHT
        self.WIDTH = train_data.WIDTH

        self.position_save_root = os.path.join(data_root, "pointmap")
        self.position_mask_save_root = os.path.join(data_root, "pointmap_mask")
        self.position_vis_save_root = os.path.join(data_root, "pointmap_vis")
        os.makedirs(self.position_save_root, exist_ok=True)
        os.makedirs(self.position_vis_save_root, exist_ok=True)    
        os.makedirs(self.position_mask_save_root, exist_ok=True)    
    
    def _load_img(self, pth):
        return transforms.ToTensor()(Image.open(pth))[None, :3, ...].to(self.device)

    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
        faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
        self.mesh = mesh
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

        # fit vertices in [0,1] bounding box
        offset = torch.mean(self.vertices, dim=1, keepdim=True)[0, 0]  # [3]
        can_vertices = self.vertices - offset
        scale = torch.max(can_vertices, dim=1)[0] - torch.min(can_vertices, dim=1)[0]
        scale = torch.max(scale)
        scale = 2 / scale
        self.canonical_offset = offset[..., None, None]
        self.canonical_scale = scale[..., None, None]
        from refinement.volume_texture import file_sha256
        with open(os.path.join(opt.save_root, 'normalization.json'), 'w') as stream:
            json.dump(dict(offset=offset.cpu().tolist(), scale=float(scale.item()),
                           mesh_sha256=file_sha256(mesh_uv_path)), stream, indent=2)
        
        # # debug usage
        # can_vertices = can_vertices * scale
        # trimesh.Trimesh(
        #     vertices=can_vertices[0].cpu().numpy(), faces=faces.cpu().numpy(),
        # ).export("debug.obj")

    def render(self):
        for data in tqdm(self.dataloader):        
            c2w = data["c2w"]
            bs = c2w.shape[0]
            cam_ext = torch.inverse(c2w)[:, :3]  # [b,3,4]
            cam_int = data["intrinsic_rel"]  # [b,3,3]
            faces = self.faces.repeat(bs, 1, 1)  # [b,v,3]
            vertices = self.vertices.repeat(bs, 1, 1)  # [b,v,3]
            img_name = os.path.splitext(data["img_name"][0])[0]
            attrs = torch.cat([
                vertices, torch.ones_like(vertices[..., :1])
            ], dim=-1)
            mesh_dict = {
                "faces": faces,
                "vertice": vertices,
                "attributes": attrs,
                "size": (self.HEIGHT, self.WIDTH),
            }
            attr_img, pix_to_face = self.mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)  # [b,2,h,w] [b,h,w,1]
            
            # scale to [-1, 1] bounding box
            point_img = attr_img[:, :3]
            mask_img = attr_img[:, 3:4]
            point_img = (point_img - self.canonical_offset) * self.canonical_scale
            
            save_image(mask_img, os.path.join(self.position_mask_save_root, "%s.png" % img_name))
            
            torch.save(point_img, os.path.join(self.position_save_root, "%s.pkl" % img_name))
            save_image((point_img + 1) / 2, os.path.join(self.position_vis_save_root, "%s.jpg" % img_name))


if __name__ == "__main__":
    ds = DiffusionSampler(opt)
    ds.render()
