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
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--syn', type=int, default=1)
parser.add_argument('--data_root', type=str, default="xxx")
parser.add_argument('--mesh_name', type=str, default="final_hack.obj")
parser.add_argument('--variant', type=str, default="")
opt, _ = parser.parse_known_args()

opt.meta_file_path = os.path.join(opt.data_root, "transforms.json")

opt.device = "cuda:%s" % opt.device


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
        self.id_root = opt.data_root
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

        image_dir = os.path.join(self.id_root, "image")
        # spec_image_dir = os.path.join(self.id_root, "spec")
        mask_dir = os.path.join(self.id_root, "mask")

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
                # spec_img = self._load_img(os.path.join(spec_image_dir, img_name))
                # self.spec_images.append(spec_img)
                img = self._load_img(os.path.join(image_dir, img_name))
                self.images.append(img)
                mask = self._load_img(os.path.join(mask_dir, img_name))
                self.masks.append(mask[:1])
            else:
                # self.spec_images.append(os.path.join(spec_image_dir, img_name))
                self.images.append(os.path.join(image_dir, img_name))
                self.masks.append(os.path.join(mask_dir, img_name))

        if self.cache_image:
            self.images = torch.stack(self.images, dim=0)  # [b,3,h,w]
            # self.spec_images = torch.stack(self.spec_images, dim=0)  # [b,3,h,w]
            self.masks = torch.stack(self.masks, dim=0)  # [b,1,h,w]
        
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
            render_mask = self.masks[index]
            # spec_img = self.spec_images[index]
        else:
            img = self._load_img(self.images[index])
            render_mask = self._load_img(self.masks[index])[:1]
            # spec_img = self._load_img(self.spec_images[index])

        c2w = self.cam2world[index]
        
        # origins, viewdirs = self._compute_ray(c2w)
        info = {
            "pixels": img,  # [3,h,w]
            "mask": render_mask,  # [1,h,w]
            # "spec": spec_img,
            "c2w": c2w,  # [4,4]
            "intrinsic": self.intrinsic,
            "intrinsic_rel": self.intrinsic_rel,
            "img_name": self.img_name_list[index],
            # "origins": origins[0],  # [3,h,w]
            # "viewdirs": viewdirs[0],  # [3,h,w]
        }
        return info


class DiffusionSampler:
    def __init__(self, opt):
        self.device = opt.device
        self.mesh_renderer_pt3d = MeshRenderer(self.device)

        # set dataset
        train_data = MetaShapeDataset(
            device=self.device, meta_file_path=opt.meta_file_path, mode="train",
        )
        self.dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
        self._load_geometry(os.path.join(opt.data_root, opt.mesh_name))

        self.HEIGHT = train_data.HEIGHT
        self.WIDTH = train_data.WIDTH
        data_root = os.path.join(opt.data_root, opt.variant)
        self.save_root = os.path.join(data_root, "uv")
        self.save_vis_root = os.path.join(data_root, "uv_vis")
        self.save_mask_root = os.path.join(data_root, "uv_mask")
        self.save_pho_mask_root = os.path.join(data_root, "pho_mask")
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.save_vis_root, exist_ok=True)
        os.makedirs(self.save_mask_root, exist_ok=True)
        os.makedirs(self.save_pho_mask_root, exist_ok=True)

        self.narrow_mask = 1 - self._load_img("assets/narrow_mask.png")
    
    def _load_img(self, pth):
        return transforms.ToTensor()(Image.open(pth))[None, :3, ...].to(self.device)

    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        uv = torch.from_numpy(mesh.visual.uv).to(device).float()  # [v,2]
        vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
        self.vert_normal = torch.from_numpy(mesh.vertex_normals).to(device).float()  # [v,3]
        faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

    def compute_rays(self, data):
        '''
        c2w: [b,4,4]
        '''
        c2w = data["c2w"]
        intrinsic = data["intrinsic"][0]
        height = self.HEIGHT
        width = self.WIDTH
        device = self.device

        x, y = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
        )
        # for pytorch 1.9 cannot specify indexing="xy"
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        x = x.flatten().to(device)
        y = y.flatten().to(device)
        camera_dirs = torch.stack(
            [
                (x - intrinsic[0, 2] + 0.5) / intrinsic[0, 0],
                (y - intrinsic[1, 2] + 0.5) / intrinsic[1, 1],
                torch.ones_like(y),
            ],
            dim=-1,
        )  # [num_rays,3]

        # transform view direction to world space
        directions = torch.matmul(c2w[:, None, :3, :3], camera_dirs[..., None])[..., 0]  # [b,num_rays,3]
        origins = c2w[:, :3, -1]  # [b,3]
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        viewdirs = torch.reshape(viewdirs, (-1, height, width, 3))  # [b,h,w,3]
        return origins, viewdirs

    def render(self):
        for data in tqdm(self.dataloader):        
            c2w = data["c2w"]
            bs = c2w.shape[0]
            cam_ext = torch.inverse(c2w)[:, :3]  # [b,3,4]
            cam_int = data["intrinsic_rel"]  # [b,3,3]
            faces = self.faces.repeat(bs, 1, 1)  # [b,v,3]
            vertices = self.vertices.repeat(bs, 1, 1)  # [b,v,3]
            uv = self.uv.repeat(bs, 1, 1)  # [b,v,2]
            # mask = torch.ones_like(uv[..., :1])  # [b,v,1]
            img_name = os.path.splitext(data["img_name"][0])[0]

            origins, _ = self.compute_rays(data)
            viewdir = origins - vertices
            mask = torch.sum(viewdir * self.vert_normal, dim=-1, keepdim=True)
            mask = (mask > 0).float()

            attrs = torch.cat([uv, mask], dim=-1)  # [b,v,c]
            mesh_dict = {
                "faces": faces,
                "vertice": vertices,
                "attributes": attrs,
                "size": (self.HEIGHT, self.WIDTH),
            }
            attr_img, uv_da = self.mesh_renderer_pt3d.render_mesh(mesh_dict, cam_int, cam_ext)  # [b,2,h,w] [b,h,w,1]
            # compute img space attrs
            uv_img = attr_img[:, :2]  # [b,2,h,w]
            mask_img = attr_img[:, 2:3]

            uv_img_vis = mask_img * (uv_img + 1) / 2
            uv_img_vis = torch.cat([uv_img_vis, torch.zeros_like(mask_img)], dim=1)
            save_image(mask_img, os.path.join(self.save_mask_root, "%s.png" % img_name))
            save_image(uv_img_vis, os.path.join(self.save_vis_root, "%s.jpg" % img_name))
            torch.save(uv_img, os.path.join(self.save_root, "%s.pkl" % img_name))

            narrow_mask = F.grid_sample(self.narrow_mask, uv_img.permute(0, 2, 3, 1))
            narrow_mask = narrow_mask * mask_img
            save_image(narrow_mask, os.path.join(self.save_pho_mask_root, "%s.png" % img_name))


if __name__ == "__main__":
    ds = DiffusionSampler(opt)
    ds.render()
