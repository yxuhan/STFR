import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--img_name', type=str, default="image")
parser.add_argument('--num_view', type=int, default=16)
parser.add_argument('--syn', type=int, default=1)
parser.add_argument('--vis_freq', type=int, default=50)
parser.add_argument('--data_root', type=str, default="xxx")
parser.add_argument('--mesh_name', type=str, default="final_hack.obj")
parser.add_argument('--variant', type=str, default="")

opt, _ = parser.parse_known_args()

opt.meta_file_path = os.path.join(opt.data_root, "transforms.json")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.device


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import math
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lpips
from torch.utils.tensorboard import SummaryWriter
import kornia
import trimesh
import open3d as o3d

from module import InstantNGPNetwork
from mesh_renderer import MeshRenderer


net_cfg = {
    "log2_hashmap_size": 16,
    "finest_level": 1024,
    "uv_reso_w": 1024,
    "uv_reso_h": 1024,
    "batch_size": 1048576,
}


def compute_vertex_visibility_o3d(o3d_scene, point, uv_vertices):
    # 创建光线张量
    directions = uv_vertices - point
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    ray_origins = np.repeat(point[None, ...], len(uv_vertices), axis=0)
    rays = np.concatenate([ray_origins, directions], axis=1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    # 计算交点个数
    intersection_counts = o3d_scene.count_intersections(rays)
    return intersection_counts.numpy()


class UVInstantNGP(nn.Module):
    '''
    map a 2d position to its uv parameters
    '''
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg["uv_reso_w"]
        self.height = cfg["uv_reso_h"]
        self.cfg = cfg

        finest_level = cfg["finest_level"]
        log2_hashmap_size = cfg["log2_hashmap_size"]
        self.diff_uv_net = InstantNGPNetwork(
            finest_level=finest_level, log2_hashmap_size=log2_hashmap_size, out_chns=3,
        )
    
    def _batch_forward(self, net):
        batch_size = self.cfg["batch_size"]
        x, y = torch.meshgrid(
            torch.arange(self.width),
            torch.arange(self.height),
            indexing="xy",
        )
        x = x.flatten().cuda() / self.width
        y = y.flatten().cuda() / self.height
        coord = torch.stack([x, y], dim=1)
        start = 0
        res_list = []
        cnt = 0
        while start < len(coord):
            cnt += 1
            end = start + batch_size
            cur_coord = coord[start:end]
            cur_res = net(cur_coord)
            res_list.append(cur_res)
            start = end
        res_list = torch.cat(res_list, dim=0)
        res_list = res_list.reshape(self.height, self.width, -1)
        return res_list.permute(2, 0, 1)
        
    def forward(self):
        diff_uv = self._batch_forward(self.diff_uv_net)        
        diff_uv = torch.sigmoid(diff_uv)
        return diff_uv[None, ...]


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
        skip_data_size = math.ceil(len(frames) / opt.num_view)
        if mode in ["val"]:
            frames = frames[::skip_data_size]
        self.frames = frames
        self.num_frames = len(self.frames)

        image_dir = os.path.join(self.id_root, opt.img_name)
        mask_dir = os.path.join(self.id_root, opt.variant, "uv_mask")
        uv_dir = os.path.join(self.id_root, opt.variant, "uv")

        # load per-frame meta information
        self.cam2world = []
        self.images, self.masks, self.uv_images = [], [], []
        self.img_idx = []
        self.img_name_list = []
        cnt = 0
        for f_id in tqdm(range(self.num_frames)):
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            if opt.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            img_name = os.path.basename(frames[f_id]['file_path'])
            uv_img_name = "%s.pkl" % os.path.splitext(img_name)[0]

            img = self._load_img(os.path.join(image_dir, img_name))
            self.images.append(img)
            mask = self._load_img(os.path.join(mask_dir, img_name))
            self.masks.append(mask[:1])
            uv = torch.load(os.path.join(uv_dir, uv_img_name), map_location=self.device)[0]
            self.uv_images.append(uv)
            self.img_idx.append(cnt)
            self.img_name_list.append(img_name)
            cnt += 1

        self.images = torch.stack(self.images, dim=0)  # [b,3,h,w]
        self.masks = torch.stack(self.masks, dim=0)  # [b,1,h,w]
        self.uv_images = torch.stack(self.uv_images, dim=0)  # [b,2,h,w]
        
        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)  # [nf,4,4]
        self.cam2world = torch.from_numpy(self.cam2world).to(self.device)  # [nf,4,4]

        print("Create [%s] dataset, total [%d] frames" % (self.mode, self.num_frames))

    def __len__(self):
        return self.num_frames
        
    def _load_img(self, pth):
        if os.path.exists(pth):
            img = transforms.ToTensor()(Image.open(pth))
            return img.to(self.device)
        else:
            return torch.zeros(3, self.HEIGHT, self.WIDTH).to(self.device)
        
    def __getitem__(self, index):
        img = self.images[index]
        render_mask = self.masks[index]
        if img.shape[0] == 4:
            render_mask = render_mask * img[3:]
            img = img[:3] * img[3:]
        
        uv_img = self.uv_images[index]

        c2w = self.cam2world[index]
        
        # origins, viewdirs = self._compute_ray(c2w)
        info = {
            "pixels": img,  # [3,h,w]
            "mask": render_mask,  # [1,h,w]
            "uv": uv_img,
            "c2w": c2w,  # [4,4]
            "intrinsic": self.intrinsic,
            "intrinsic_rel": self.intrinsic_rel,
            "idx": self.img_idx[index],
            "img_name": self.img_name_list[index],
            # "origins": origins[0],  # [3,h,w]
            # "viewdirs": viewdirs[0],  # [3,h,w]
        }
        return info


class DiffusionSampler:
    def __init__(self, opt):
        self.device = "cuda"
        self.log_dir = os.path.join(opt.data_root, opt.variant, "texture", opt.img_name)
        self.gamma = 2.2

        # set dataset
        train_data = MetaShapeDataset(
            device="cpu", meta_file_path=opt.meta_file_path, mode="val",
        )
        self.batch_size = 4
        
        self.dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)

        self.HEIGHT = train_data.HEIGHT  # image height
        self.WIDTH = train_data.WIDTH  # image width
        self.weight_loss_tau = 100

        self.uv_reso_h = net_cfg["uv_reso_h"]
        self.uv_reso_w = net_cfg["uv_reso_w"]

        self.network = UVInstantNGP(net_cfg)
        self.network = self.network.to(self.device)

        self.lpips_loss = lpips.LPIPS(net="vgg").cuda()
        self.lpips_loss.requires_grad_(False)

        self.mesh_renderer = MeshRenderer(self.device)
        data_root = os.path.dirname(opt.meta_file_path)
        self.mesh_path = os.path.join(data_root, opt.mesh_name)
        self._load_geometry(self.mesh_path)

        self.loss_weight_img = self.compute_loss_weight()
    
    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        uv = torch.from_numpy(mesh.visual.uv).to(device).float()  # [v,2]
        vertices = torch.from_numpy(mesh.vertices).to(device).float()  # [v,3]
        normal = torch.from_numpy(mesh.vertex_normals).to(device).float()  # [v,3]
        normal = F.normalize(normal, dim=-1)
        faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

        attrs = torch.cat([
            normal[None, ...], vertices[None, ...],
        ], dim=-1)

        # render the normal in UV space as the initialization
        uv_mesh_dict = {
            "faces": self.faces,
            "vertice": torch.cat([self.uv, torch.ones_like(self.uv[..., :1])], dim=-1),  # [1,v,3]
            "attributes": attrs,  # [1,v,3]
            "size": (self.uv_reso_h, self.uv_reso_w),
        }
        attrs_uv, uv_pix_to_face = self.mesh_renderer.render_ndc(uv_mesh_dict)  # [1,6,uv,uv]
        
        normal_uv = attrs_uv[:, :3]
        pos_uv = attrs_uv[:, 3:6]
        self.object_normal_lr_4k = torch.clone(normal_uv)  # [1,3,uh,uw]
        self.position_uv = torch.clone(pos_uv)  # [1,1,uh,uw]
        self.uv_nonuse_mask = (uv_pix_to_face == -1).float().permute(0, 3, 1, 2)  # [1,1,uh,uw]

    def compute_rays(self, data):
        '''
        c2w: [b,4,4]
        '''
        c2w = data["c2w"]
        intrinsic = data["intrinsic"][0]
        height = self.HEIGHT
        width = self.WIDTH
        device = c2w.device

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

    def render_in_uv(self, data, shading):
        for k in data:
            try:
                data[k] = data[k].to(self.device)
            except:
                pass
        
        c2w = data["c2w"]
        uv_img = data["uv"]
        uv_img = uv_img.permute(0, 2, 3, 1).contiguous()
        
        tex_img = F.grid_sample(
            shading.permute(0, 3, 1, 2).repeat(len(uv_img), 1, 1, 1), 
            uv_img, 
            align_corners=True
        )  # [b,3,h,w]   

        return tex_img

    def compute_loss_weight(self):
        uv_viewdir = []
        uv_list = []
        vis_list = []
        verts_uv_np = self.position_uv[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        cnt = 0
        ks = 15
        kernel = torch.ones(ks, ks).to(self.device)

        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_legacy)
        for data in tqdm(self.val_dataloader):
            for k in data:
                try:
                    data[k] = data[k].to(self.device)
                except:
                    pass
            origins, viewdirs = self.compute_rays(data)
            origins_np = origins[0].cpu().numpy()
            
            vis_mask = compute_vertex_visibility_o3d(
                o3d_scene=scene, point=origins_np, uv_vertices=verts_uv_np,
            )
            vis_mask_torch = torch.from_numpy(vis_mask).float().to(self.device).reshape(1, 1, self.uv_reso_h, self.uv_reso_w)
            vis_mask_torch = (vis_mask_torch <= 1).float()

            vis_mask_torch = kornia.morphology.erosion(vis_mask_torch, kernel=kernel)
            vis_list.append(vis_mask_torch)
            # save_image(vis_mask_torch, os.path.join("workspace/weight_img_vis_erode", "%05d.jpg" % cnt))
            cnt += 1
            cur_uv_viewdir = F.normalize(self.position_uv - origins[..., None, None], dim=1)
            uv_viewdir.append(cur_uv_viewdir)
            uv_list.append(data["uv"])
        uv_viewdir = torch.cat(uv_viewdir, dim=0)
        uv_list = torch.cat(uv_list, dim=0)

        weight = torch.sum(-self.object_normal_lr_4k * uv_viewdir, dim=-3, keepdim=True)
        weight = torch.clamp(weight, min=0.)  # [nview,npatch,1,uh,uw]
        vis_list = torch.cat(vis_list, dim=0)
        weight = weight * vis_list
        # save_image(vis_list, "vis_uv.jpg")

        # softmax weight controlled by tau
        max_weight = F.softmax(self.weight_loss_tau * weight, dim=0)
        # save_image(max_weight, "weight.jpg")

        # for i in range(len(max_weight)):
        #     save_image(max_weight[i], os.path.join("workspace/weight_img_vis_erode", "%05d_weight.jpg" % i))

        uv_list = uv_list.permute(0, 2, 3, 1).contiguous()
        weight_img = F.grid_sample(max_weight, uv_list)
        return weight_img

    def optimize(self):
        params = [{"params": self.network.parameters()}]
        lr = 0.01
        optimizer = torch.optim.Adam(params=params, lr=lr)
        grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)
        writer = SummaryWriter(self.log_dir)
        
        for i in tqdm(range(151)):
            for data in self.dataloader:
                uv_shading = self.network().permute(0, 2, 3, 1).contiguous()
                render = self.render_in_uv(data, uv_shading)
                weight_img = self.loss_weight_img[data["idx"]]

                mask = data["mask"]
                gt = data["pixels"]

                img_pred = render * mask
                img_gt = gt * mask

                grad_gt = kornia.filters.spatial_gradient(img_gt)
                grad_pred = kornia.filters.spatial_gradient(img_pred)
                loss_l1 = F.l1_loss(grad_pred * weight_img[:, :, None, ...], grad_gt * weight_img[:, :, None, ...])
                
                if i > 100:
                    loss_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
                    loss = 10 * loss_l1 + 0.1 * loss_lpips
                else:
                    loss = 10 * loss_l1

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                writer.add_scalar('L1 Loss', loss_l1.item(), i)

            if i % opt.vis_freq == 0:
                cur_log_dir = os.path.join(self.log_dir, "%05d" % i)
                os.makedirs(cur_log_dir, exist_ok=True)
                with torch.no_grad():
                    render_list = []
                    gt_list = []
                    psnr_list = []
                    ssim_list = []
                    lpips_list = []
                    img_name_list = []
                    for data in self.val_dataloader:
                        render = self.render_in_uv(data, uv_shading)
                        mask = data["mask"]
                        gt = data["pixels"]
                        img_pred = render * mask
                        img_gt = gt * mask
                        gt_list.append(img_gt)
                        render_list.append(img_pred)
                        cur_psnr = kornia.metrics.psnr(img_pred, img_gt, max_val=1.).item()
                        cur_ssim = kornia.metrics.ssim(img_pred, img_gt, window_size=3).mean().item()
                        cur_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
                        psnr_list.append(cur_psnr)
                        ssim_list.append(cur_ssim)
                        lpips_list.append(cur_lpips)
                        img_name_list.append(data["img_name"][0])
                    
                    render_list = torch.cat(render_list, dim=0)
                    gt_list = torch.cat(gt_list, dim=0)
                    save_image(render_list, os.path.join(cur_log_dir, "render.jpg"))
                    save_image(gt_list, os.path.join(cur_log_dir, "gt.jpg"))
                    save_image(uv_shading.permute(0, 3, 1, 2), os.path.join(cur_log_dir, "uv.png"))
                    save_image(uv_shading.permute(0, 3, 1, 2), os.path.join(cur_log_dir, "uv_vis.png"))

                    psnr_score = sum(psnr_list) / len(psnr_list)
                    ssim_score = sum(ssim_list) / len(ssim_list)
                    lpips_score = sum(lpips_list) / len(lpips_list)
                    writer.add_scalar('ssim', ssim_score, i)
                    writer.add_scalar('psnr', psnr_score, i)
                    writer.add_scalar('lpips', lpips_score, i)
                    print("[iter %05d][PSNR %.4f][SSIM %.4f][LPIPS %.4f]" % (
                        i, psnr_score, ssim_score, lpips_score,
                    ))


if __name__ == "__main__":
    ir = DiffusionSampler(opt)
    ir.optimize()
