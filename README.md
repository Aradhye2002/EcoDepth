<div align="center">
<h1>ECoDepth v2.0.0</h1>

**CVPR 2024**  
<a href='https://ecodepth-iitd.github.io' style="margin-right: 20px;"><img src='https://img.shields.io/badge/Project Page-ECoDepth-darkgreen' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2403.18807" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-arXiv-maroon' alt='arXiv page'></a>
<a href="https://arxiv.org/abs/2403.18807" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-CvF-blue' alt='IEEE Xplore Paper'></a>
<a href="https://arxiv.org/abs/2403.18807" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Supplementary-CvF-blue' alt='IEEE Xplore Paper'></a>

[Suraj Patni](https://github.com/surajiitd)\*,
[Aradhye Agarwal](https://github.com/Aradhye2002)\*,
[Chetan Arora](https://www.cse.iitd.ac.in/~chetan)<br/>

</div>

Welcome to the **restructured codebase** for **ECoDepth**, our official implementation for monocular depth estimation (MDE) as presented in our CVPR 2024 paper. This repository has been significantly reorganized to improve usability, readability, and extensibility. 

> **Important:** The original code used to generate the results in our paper is tagged as [v1.0.0](https://github.com/Aradhye2002/EcoDepth/releases/tag/v1.0.0), which you can download from the Releases section. For most practical purposes—such as training on custom datasets or performing inference—**we strongly recommend using the new [v2.0.0](https://github.com/Aradhye2002/EcoDepth/releases/tag/v2.0.0)** outlined here.


## News
- **[April 2024] Inference scripts for video or image to depth.**
- [March 2024] Pretrained checkpoints for NYUv2 and KITTI datasets.
- [March 2024] Training and Evaluation code released!
- [Feb 2024] ECoDepth accepted in CVPR'2024.

## Table of Contents
1. [Overview of v2.0.0 Improvements](#overview-of-v200-improvements)
2. [Setup](#setup)
3. [Dataset Download (NYU Depth V2)](#dataset-download-nyu-depth-v2)
4. [DepthDataset API](#depthdataset-api)
5. [EcoDepth Model API](#ecodepth-model-api)
6. [Training Workflow](#training-workflow)
7. [Testing Workflow](#testing-workflow)
8. [Inference Workflow](#inference-workflow)
9. [Citation](#citation)



## Overview of v2.0.0 Improvements

1. **Integrated Model Downloading**  
   - In the previous version (v1.0.0), you had to manually download our checkpoints from Google Drive and place them in the correct directory. Now, the model is automatically downloaded and cached on the first run in `EcoDepth/checkpoints`. Subsequent runs will use the cached checkpoints automatically.

2. **Generic DepthDataset Module**  
   - We provide a new, flexible `DepthDataset` module that loads any custom dataset for MDE training. This was a frequent feature request. Detailed usage is given in the [DepthDataset API](#depthdataset-api) section.

3. **PyTorch Lightning Integration**  
   - The `EcoDepth` model is now a subclass of `LightningModule`, allowing for streamlined training and inference workflows via PyTorch Lightning. This also makes it straightforward to export models to ONNX or TorchScript for production use.

4. **Config-Based Workflows**  
   - We replaced bash scripts with user-friendly JSON configs, making it clearer to specify training, testing, and inference parameters.

5. **Reduced Dependencies & Simplified Setup**  
   - We removed the requirement to install the entire Stable Diffusion pipeline and numerous large CLIP or VIT models separately. Our checkpoints already contain the necessary weights, so only **one** model download is required.  
   - Dependencies like `mmcv`, which can be cumbersome to install, are no longer necessary. Installation is now simpler and more flexible.

6. **Separate Workflows**  
   - The code is structured into three main directories:  
     - `train/` for training  
     - `test/` for testing  
     - `infer/` for inference  
   - Each directory contains its own config files, making each workflow highly modular.



## Setup

1. **Install PyTorch (with or without GPU support)**  
   - Refer to the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) for commands tailored to your environment.  
   - Example (with CUDA 12.4):
     ```bash
     conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
     ```

2. **Install python3 Dependencies**  
   - From the repository’s root directory, run:
     ```bash
     pip install -r requirements.txt
     ```
   - We have **not** pinned specific versions to reduce potential conflicts. Let the dependency resolver pick suitable versions for your system.

3. **(Optional) Download NYU Depth V2 Dataset**  
   - If you plan to train on the NYU Depth V2 dataset, simply run:
     ```bash
     bash download_nyu.sh
     ```
   - This downloads and unzips the dataset from [HF dataset `aradhye/nyu_depth_v2`](https://huggingface.co/datasets/aradhye/nyu_depth_v2) into a directory named `nyu_depth_v2` under `datasets`. The filenames are already provided as text files under `filenames/nyu_depth_v2`.



## Dataset Download (NYU Depth V2)

If you want to replicate our NYU Depth V2 experiments:

1. Run:
   ```bash
   bash download_nyu.sh
   ```
2. This script:
   - Downloads NYU Depth V2 from [aradhye/nyu_depth_v2](https://huggingface.co/datasets/aradhye/nyu_depth_v2).
   - Unzips the dataset in `datasets/nyu_depth_v2/`.
   - Provides file lists in `filenames/nyu_depth_v2/`.

You can then set the corresponding paths in your JSON configs (see the [Training Workflow](#training-workflow) section).



## DepthDataset API

`DepthDataset` is a generic dataset class designed for pairs of RGB images and depth maps. It requires an `args` object (which can be a namespace or a dictionary) with the following attributes:

- **`is_train` (bool)**  
  Indicates whether the dataset is used for training (`True`) or evaluation/testing (`False`). Some augmentations (e.g., random cropping) are only applied in training mode.

- **`filenames_path` (str)**  
  Path to a text file containing pairs of image and depth map paths, separated by a space. 

- **`data_path` (str)**  
  A directory path that is prepended to each filename from `filenames_path`. Thus, the actual file loaded is `data_path + path_in_filenames`.

- **`depth_factor` (float)**  
  Divides the raw depth values to convert them into meters. For NYU Depth V2, `depth_factor=1000.0`; for KITTI, `depth_factor=256.0`.

- **`do_random_crop` (bool)**  
  Whether to perform random cropping on the image/depth pairs (only if `is_train=True`). If `do_random_crop` is `True`, you must also set:
  - **`crop_h` (int)**: Crop height  
  - **`crop_w` (int)**: Crop width  

  If images are smaller than `crop_h`×`crop_w`, zero padding is applied first.

- **`use_cut_depth` (bool)**  
  Whether to use [CutDepth](https://arxiv.org/abs/2107.07684) to reduce overfitting. We found it helpful for indoor datasets (e.g., NYU Depth V2) but not for outdoor datasets (e.g., KITTI). Only used during training.



## EcoDepth Model API

`EcoDepth` is implemented as a subclass of PyTorch Lightning’s `LightningModule`. The constructor expects an `args` object with these key attributes:

- **`train_from_scratch` (bool)**  
  Currently should always be `False`. To train from scratch, you would need the base pretrained weights. Typically, you will **finetune** using our published checkpoints.

- **`eval_crop` (str)**  
  Determines the evaluation cropping strategy. Possible values:
  - `"eigen"`: Used for NYU  
  - `"garg"`: Used for KITTI  
  - `"custom"`: Implement your own function in `utils.py` and set `eval_crop="custom"`.  
  - `"none"`: No cropping

- **`no_of_classes` (int)**  
  Number of scene classes for internal embeddings. For NYU (indoor model) use `100`; for VKITTI (outdoor model) use `200`.

- **`max_depth` (float)**  
  Maximum depth value the model will predict. Typically:
  - `10.0` for indoor (NYU)  
  - `80.0` for outdoor (KITTI)



## Training Workflow

1. **Navigate to `train/`**  
2. **Edit `train_config.json`**  
   - **Data arguments**:  
     - `train_filenames_path`, `train_data_path`, `train_depth_factor`  
     - `test_filenames_path`, `test_data_path`, `test_depth_factor`  
   - **Model arguments**:  
     - `eval_crop`, `no_of_classes`, `max_depth`  
   - **Training arguments**:  
     - `ckpt_path`: Path to a Lightning checkpoint (for finetuning/resuming). If this is an empty string, you must specify `scene="indoor"` or `"outdoor"`, triggering automatic model download.  
     - `epochs`: Total training epochs  
     - `weight_decay`, `lr`: Optimizer hyperparameters  
     - `val_check_interval`: Validation frequency (in training steps)  

3. **Run Training**  
   ```bash
   python3 train.py
   ```
   PyTorch Lightning will handle checkpointing automatically (by default, in `train/lightning_logs/`).



## Testing Workflow

1. **Navigate to `test/`**  
2. **Edit `test_config.json`**  
   - Similar to training config, but no training arguments.  
   - Point `ckpt_path` to the checkpoint you want to evaluate, or leave empty if you want to use the provided models.  
3. **Run Testing**  
   ```bash
   python3 test.py
   ```
   This script reports evaluation metrics (e.g., RMSE, MAE, δ thresholds).



## Inference Workflow

There are two scripts provided in the `infer/` directory: one for images (`infer_image.py`) and one for videos (`infer_video.py`).

### Image Inference

1. **Navigate to `infer/`**  
2. **Edit `image_config.json`**  
   - **Key arguments**:  
     - `image_path`: Path to a single image or a directory containing multiple images (recursively processed).  
     - `outdir`: Output directory for predicted depth maps.  
     - `resolution`: Scale factor for processing images (higher resolution => more GPU memory usage).  
     - `flip_test` (bool): Whether to perform horizontal flip as a test-time augmentation.  
     - `grayscale` (bool): Output in grayscale (if `false`, uses a colorized depth map).  
     - `pred_only` (bool): Whether to output **only** the depth map.

3. **Run Image Inference**  
   ```bash
   python3 infer_image.py
   ```
   Results are written to `outdir`, preserving subdirectory structure relative to `image_path`.

### Video Inference

1. **Edit `video_config.json`**  
   - **Key arguments**:  
     - `video_path`: Path to the video file.  
     - `outdir`: Output directory for frames or depth predictions.  
     - `vmax`: Depth values are clipped to this maximum.  
2. **Run Video Inference**  
   ```bash
   python3 infer_video.py
   ```



## Citation

If you find **ECoDepth** helpful in your research or work, please cite our CVPR 2024 paper:

```
@InProceedings{Patni_2024_CVPR,
  author    = {Patni, Suraj and Agarwal, Aradhye and Arora, Chetan},
  title     = {ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {28285-28295}
}
```



**Thank you for using ECoDepth!**  
For any questions or suggestions, feel free to open an issue. We hope this restructured codebase helps you train on custom datasets and perform fast, efficient inference.