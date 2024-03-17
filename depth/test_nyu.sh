export CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nproc_per_node=1 \
    test.py \
    --dataset nyudepthv2 --data_path ./ \
    --max_depth 10.0 --max_depth_eval 10.0 \
    --num_filters 32 32 32 --deconv_kernels 2 2 2\
    --flip_test  --ckpt_dir $1 \
    --crop_h 480 --crop_w 480 \
    --vit_model google/vit-base-patch16-224 \
    --save_visualize --save_eval_pngs \
    --latent_scaling \
    --reg_loss_abs_of_embed False \
    --exp_name testing_nyu \
    --num_of_diffusion_step 1 \
    ${@:2} \