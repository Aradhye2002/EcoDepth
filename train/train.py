import sys
sys.path.append("..")
from dataset import DepthDataset
import json
from torch.utils.data import DataLoader
from model import EcoDepth
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from utils import download_model

class Args:
    def __init__(self):
        with open("train_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

model = EcoDepth(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    version = "ecodepthv2" if args.enable_v2 else "ecodepth"
    download_model(model_str, version)
    args.ckpt_path = f"../checkpoints/{model_str}"

model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"])

train_dataset = DepthDataset(
    args=args, 
    is_train=True, 
    filenames_path=args.train_filenames_path, 
    data_path=args.train_data_path, 
    depth_factor=args.train_depth_factor
)

train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

val_dataset = DepthDataset(
    args=args, 
    is_train=False, 
    filenames_path=args.val_filenames_path, 
    data_path=args.val_data_path, 
    depth_factor=args.val_depth_factor
)

val_loader = DataLoader(val_dataset, num_workers=args.num_workers)

checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    save_last=True,
    monitor="val_loss",
    save_weights_only=True,
)

trainer = L.Trainer(
    max_epochs=args.epochs,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback]
)

trainer.fit(
    model=model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader,
)
