export CUDA_VISIBLE_DEVICES=0

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python infer.py \
    --img_path path/to/png \
    --video_path path/to/mp4 \
    --flip_test \
    --max_depth 80.0 \
    --min_depth 1e-3 \
    --no_of_classes 200 \
    --ckpt_dir checkpoints/vkitti.ckpt \