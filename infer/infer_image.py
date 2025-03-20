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
        with open("image_config.json", "r") as f:
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
    download_model(model_str)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"])

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(DEVICE).eval()

if os.path.isfile(args.image_path):
    if args.video_path.endswith('txt'):
        with open(args.image_path, 'r') as f:
            lines = f.read().splitlines()
    else:
        filenames = [args.image_path]
else:
    filenames = glob.glob(os.path.join(args.image_path, '**/*'), recursive=True)

os.makedirs(args.outdir, exist_ok=True)

margin_width = 50
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

for k, filename in enumerate(filenames):
    print(f'Progress {k+1}/{len(filenames)}: {filename}')
    
    raw_frame = cv2.imread(filename)
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
    frame_height, frame_width = frame.shape[:2]
    
    curr_area = frame_width * frame_height
    scale = math.sqrt(base_area / curr_area)
    new_w, new_h = int(scale * frame_width), int(scale * frame_height)
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + margin_width
    
    output_path = os.path.join(args.outdir, 'depth_' + os.path.splitext(os.path.basename(filename))[0] + '.png')
    
    width_adjustment = 0
    height_adjustment = int(40/scale)
    
        
    frame = cv2.resize(frame, (new_w, new_h))
    
    frame = transforms.ToTensor()(frame).unsqueeze(0)
    
    if args.flip_test:
        frame = torch.cat([frame, frame.flip(-1)])
    
    frame = frame.to(DEVICE)
    
    with torch.no_grad():
        depth = model(frame)
    if args.flip_test:
        depth = ((depth[0] + depth[1].flip(-1))/2).unsqueeze(0)
    
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth[0, 0].detach().cpu().numpy().astype(np.uint8)
    
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    depth = cv2.resize(depth, (frame_width, frame_height))
    
    if args.pred_only:
        cv2.imwrite(output_path, depth)
    else:
        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
        combined_frame = cv2.hconcat([raw_frame, split_region, depth])
        combined_frame = combined_frame[height_adjustment:, width_adjustment:]
        cv2.imwrite(output_path, combined_frame)
