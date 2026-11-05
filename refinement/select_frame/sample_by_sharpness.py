import os
import argparse
import torch
import math
import json
import shutil


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_root', type=str)
parser.add_argument('--cam_path', type=str)
parser.add_argument('--save_root', type=str)
parser.add_argument('--num_view', type=int, default=16)
opt, _ = parser.parse_known_args()


meta_file_path = opt.cam_path
with open(meta_file_path, 'r') as f:
    meta = json.load(f)

frames = meta["frames"]
frames = sorted(frames, key=lambda d: d['file_path'])
skip_data_size = math.ceil(len(frames) / opt.num_view)
new_frames = []

interval = skip_data_size

raw_frame_root = opt.img_root
img_root = os.path.join(opt.save_root, "image")
os.makedirs(img_root, exist_ok=True)
sharp_info = torch.load(os.path.join(opt.save_root, "sharpness.pkl"))


img_pth_list = sorted(os.listdir(raw_frame_root))
num_img = len(img_pth_list)
left = 0
while left + 1 < num_img:
    right = min(left + interval, num_img)
    max_sharp = 0
    for i in range(left, right):
        cur_pth = img_pth_list[i]
        cur_sharp = sharp_info[cur_pth] 
        if cur_sharp > max_sharp:
            max_sharp = cur_sharp
            max_path = cur_pth
            max_idx = i
    shutil.copy2(os.path.join(raw_frame_root, max_path), os.path.join(img_root, max_path))
    new_frames.append(frames[max_idx])

    left = right

meta["frames"] = new_frames
with open(os.path.join(opt.save_root, "select_sharp.json"), "w") as outfile:
    json.dump(meta, outfile, indent=4)
