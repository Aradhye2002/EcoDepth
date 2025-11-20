# Code adapted from the Depth-Anything-V2 repo: 
# https://github.com/DepthAnything/Depth-Anything-V2
import sys
sys.path.append("..")
import json
from model import EcoDepth
import torch
import torchvision.transforms as transforms
import cv2
import glob
import matplotlib
import numpy as np
import os
import math
from utils import download_model

class Args:
    def __init__(self):
        with open("video_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

# For compatibility
args.eval_crop = "no_crop"

base_area = args.resolution * 480 * 640

model = EcoDepth(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    version = "ecodepthv2" if args.enable_v2 else "ecodepth"
    download_model(model_str, version)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"])

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(DEVICE).eval()

if os.path.isfile(args.video_path):
    if args.video_path.endswith('txt'):
        with open(args.video_path, 'r') as f:
            lines = f.read().splitlines()
    else:
        filenames = [args.video_path]
else:
    filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

os.makedirs(args.outdir, exist_ok=True)

margin_width = 50
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

for k, filename in enumerate(filenames):
    print(f'Progress {k+1}/{len(filenames)}: {filename}')
    
    raw_video = cv2.VideoCapture(filename)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

    curr_area = frame_width * frame_height
    scale = math.sqrt(base_area / curr_area)
    new_w, new_h = int(scale * frame_width), int(scale * frame_height)
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + margin_width
    
    output_path = os.path.join(args.outdir, 'depth_' + os.path.splitext(os.path.basename(filename))[0] + '.mp4')
    
    width_adjustment = 0
    height_adjustment = int(40/scale)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width - 2 * width_adjustment, frame_height - height_adjustment))
    
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        frame = cv2.resize(raw_frame, (new_w, new_h))
        
        frame = transforms.ToTensor()(frame).unsqueeze(0)
        
        if args.flip_test:
            frame = torch.cat([frame, frame.flip(-1)])
        
        frame = frame.to(DEVICE)
        
        with torch.no_grad():
            depth = model(frame)
        if args.flip_test:
            depth = ((depth[0] + depth[1].flip(-1))/2).unsqueeze(0)
        
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth[depth > args.vmax] = args.vmax
        depth = depth / args.vmax * 255.0
        depth = depth[0, 0].detach().cpu().numpy().astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        depth = cv2.resize(depth, (frame_width, frame_height))
        if args.pred_only:
            out.write(depth)
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])
            combined_frame = combined_frame[height_adjustment:, width_adjustment:]
            out.write(combined_frame)
    
    raw_video.release()
    out.release()