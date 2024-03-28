<div align="center">
<h1>ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation</h1>

**CVPR 2024**  
<a href='https://ecodepth-iitd.github.io/' style="margin-right: 20px;"><img src='https://img.shields.io/badge/Project Page-ECoDepth-darkgreen' alt='Project Page'></a>
<a href="https://arxiv.org/abs" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-arXiv-maroon' alt='arXiv page'></a>
<a href="https://arxiv.org/abs" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-CvF-blue' alt='IEEE Xplore Paper'></a>

[Suraj Patni](https://github.com/surajiitd)\*,
[Aradhye Agarwal](https://github.com/Aradhye2002)\*,
[Chetan Arora](https://www.cse.iitd.ac.in/~chetan)<br/>
\* equal contribution

</div>

<!-- display pdf -->

![Architecture Diagram](figs/aarch_diagram.png)


## News
- [Coming soon] Pretrained checkpoints for NYUv2 and KITTI datasets.
- [March 2024] Training and Inference code released!
- [Feb 2024] ECoDepth accepted in CVPR'2024.


## Installation

``` bash
git clone https://github.com/Aradhye2002/EcoDepth
cd EcoDepth
conda env create -f env.yml
conda activate ecodepth
```

## Pretrained Models

Please download the pretrained weights form [this link]() and save `.ckpt` inside `<repo root>/depth/checkpoints` directory.

## Evaluation
To evaluate our performance on NYUv2 and KITTI datasets, use `test.py` file. The trained models are publicly available, download the models using [above links](#pretrained-models).

1. **NYUv2**:  
`bash test_nyu.sh <path_to_saved_model_of_NYU>`  

2. **KITTI**:  
`bash test_kitti.sh <path_to_saved_model_of_KITTI>`

## Training 
We trained our models on 32 batch size using 8xNVIDIA A100 GPUs. Please set the `NPROC_PER_NODE` variable and `--batch_size` argument to set the batch size. We set them as `NPROC_PER_NODE=8` and `--batch_size=4`. So our effective batch_size is 32.  

1. **NYUv2**:  
`bash train_nyu.sh`  

2. **KITTI**:  
`bash train_kitti.sh`


### BibTeX (Citation)
``` bibtex
TODO
```