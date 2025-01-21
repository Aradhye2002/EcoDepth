import matplotlib.cm as cm
import numpy as np
import torch.nn.functional as F
import torch
import random
import os
import urllib.request as request
import progressbar


#################### Seeding ####################

def seed_everything(seed=42):
    """
    For REPRODUCIBILITY 
    Official source: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # To NOT randomly choose which algo to use for CUDNN operations like convolution,etc.
    torch.backends.cudnn.benchmark = False # benchmark=True will improve training performance but will lose reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)

#################### Visualization ####################

def colorize_depth(depth: np.ndarray, cmap: str ="inferno_r", vmin: float = None, vmax: float = None):
    # depth must be a numpy array of shape (h, w)
    vmin = np.min(depth) if vmin is None else vmin
    vmax = np.max(depth) if vmax is None else vmax
    depth = np.clip(depth, vmin, vmax)
    depth[0, 0] = vmin
    depth[-1, -1] = vmax
    depth[depth == 0] = vmax
    colormap = cm.get_cmap(cmap)
    colorized_depth = colormap((depth-vmin)/(vmax-vmin))
    colorized_depth = (colorized_depth[:, :, :3] * 255).astype(np.uint8)
    return colorized_depth

#################### Padding ####################

def pad(image, desired_multiple_of):
    old_h, old_w = image.shape[-2:]
    pad_h = (desired_multiple_of - old_h % desired_multiple_of) % desired_multiple_of
    pad_w = (desired_multiple_of - old_w % desired_multiple_of) % desired_multiple_of
    padding = (0, pad_w, 0, pad_h)
    image = F.pad(image, padding)
    return image, (pad_w, pad_h)

def unpad(image, padding):
    h, w = image.shape[-2:]
    pad_w, pad_h = padding
    image = image[..., :h-pad_h, :w-pad_w]
    return image

#################### Criterion ####################

def silog(pred, target):
    # Note that this assumes that the target always has some finite depth (or zero otherwise)
    # Also prediction should never be exactly zero
    # This could happen for example if your are regressing the depth and then applying a hard relu
    valid_mask = (target > 0).detach()
    diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
    loss = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.85 * torch.pow(diff_log.mean(), 2))
    return loss

#################### Crops ####################

def eigen_crop(depth):
    valid_mask = torch.zeros_like(depth)
    valid_mask[..., 45:471, 41:601] = 1
    # Set depth outside of valid region as invalid
    depth[valid_mask == 0] = 0
    return depth

def garg_crop(depth):
    h, w = depth.shape[-2:]
    valid_mask = torch.zeros_like(depth)
    valid_mask[..., int(0.40810811 * h):int(0.99189189 * h),
        int(0.03594771 * w):int(0.96405229 * w)] = 1
    depth[valid_mask == 0] = 0
    return depth

def custom_crop(depth):
    raise NotImplementedError

def no_crop(depth):
    return depth

class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_model(model_str):
    os.makedirs("../checkpoints", exist_ok=True)
    save_path = f"../checkpoints/{model_str}"
    url = f"https://huggingface.co/aradhye/ecodepth/resolve/main/{model_str}"
    if not os.path.isfile(save_path):
        print(f"Downloading {model_str}")
        request.urlretrieve(url, save_path, MyProgressBar())
    else:
        print(f"{model_str} is already downloaded")
        

    
    