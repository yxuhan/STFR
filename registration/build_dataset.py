import os
import argparse
import json
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="/root/autodl-tmp/HyxFacePipeline/Release/workspace/0227")
parser.add_argument('--device', type=str, default="1")
opt, _ = parser.parse_known_args()

opt.save_root = os.path.join(opt.data_root, "sample_dataset")

os.makedirs(opt.save_root, exist_ok=True)

meta_file_path = os.path.join(opt.data_root, "register/fine_align/align_canonical.json")
reg_geo_path = os.path.join(opt.data_root, "register/wrap/final_hack.obj")
sharp_frame_root = os.path.join(opt.data_root, "refinement/sample/image")

shutil.copytree(sharp_frame_root, os.path.join(opt.save_root, 'image'), dirs_exist_ok=True)
shutil.copy(reg_geo_path, os.path.join(opt.save_root, "final_hack.obj"))

with open(meta_file_path, 'r') as f:
    meta = json.load(f)

frames = meta["frames"]
frames = sorted(frames, key=lambda d: d['file_path'])
new_frames = []

for idx in range(len(frames)):
    img_name = os.path.basename(frames[idx]['file_path'])
    if img_name in os.listdir(sharp_frame_root):
        new_frames.append(frames[idx])

meta["frames"] = new_frames
with open(os.path.join(opt.save_root, "transforms.json"), "w") as outfile:
    json.dump(meta, outfile, indent=4)
