export CUDA_VISIBLE_DEVICES=0

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python infer.py \
    --img_path path/to/img \
    --video_path path/to/video \
    --max_depth 10.0 \
    --min_depth 1e-3 \
    --ckpt_dir checkpoints/nyu.ckpt \