export CUDA_VISIBLE_DEVICES=6
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nproc_per_node=1 \
    test.py \
    --dataset nyudepthv2 --data_path ./data/nyu \
    --max_depth 10.0 --max_depth_eval 10.0 \
    --num_filters 32 32 32 --deconv_kernels 2 2 2\
    --flip_test  --ckpt_dir $1 \
    --batch_size 1 \
    --vit_model google/vit-base-patch16-224 \
    --save_visualize --save_eval_pngs \
    --exp_name testing_nyu \
    ${@:2} \