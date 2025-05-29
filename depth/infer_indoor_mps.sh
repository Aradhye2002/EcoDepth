export mps=0

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python infer_mps.py \
    --video_path /%%%%.mp4 \
    --max_depth 10.0 \
    --min_depth 1e-3 \
    --ckpt_dir /%%%%/EcoDepth/depth/checkpoints/nyu.ckpt \