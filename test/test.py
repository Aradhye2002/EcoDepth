import sys
sys.path.append("..")
from dataset import DepthDataset
import json
from torch.utils.data import DataLoader
from model import EcoDepth
import lightning as L
import torch
from utils import download_model
class Args:
    def __init__(self):
        with open("test_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

model = EcoDepth(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    download_model(model_str)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"])

test_dataset = DepthDataset(
    args=args, 
    is_train=False, 
    filenames_path=args.test_filenames_path, 
    data_path=args.test_data_path, 
    depth_factor=args.test_depth_factor
)

test_loader = DataLoader(test_dataset, num_workers=args.num_workers)

trainer = L.Trainer(logger=False)

trainer.test(model, dataloaders=test_loader)