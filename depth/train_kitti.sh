NPROC_PER_NODE=8
NO_CPU_CORES=64

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$((NO_CPU_CORES / NPROC_PER_NODE))
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nproc_per_node=${NPROC_PER_NODE} \
    train.py \
    --batch_size 4 \
    --dataset kitti \
    --data_path /home/suraj/scratch/Datasets/kitti_depth_from_ash \
    --weight_decay 0.1 \
    --crop_h 480 \
    --crop_w 640 \
    --num_filters 32 32 32 \
    --deconv_kernels 2 2 2 \
    --flip_test \
    --layer_decay 0.9 \
    --drop_path_rate 0.3 \
    --log_dir log_dir_vision04 \
    ${@:2} \
    --exp_name kitti \
    --pro_bar True \
    --save_depths_gray \
    --save_depths_color \
    --vit_model google/vit-base-patch16-224 \
    --learning_rate_schedule one_cycle \
    --reg_loss_abs_of_embed False \
    --gradient_accumulation False \
    --num_of_diffusion_step 1 \
    --epochs 25 \
    --vit_trainable False \
    --use_cross_attn_of_unet False \
    --log_in_wandb True \
    --min_depth_eval 1e-3 \
    --max_depth_eval 80 \
    --max_depth 80 \
    --kitti_crop garg_crop \
    --do_kb_crop True \
    --cutflip False \
    --kitti_split_to_half  \
    --use_right False \
    --finetune_on_another_dataset False \
    --trainable_bins False \
    --bins_from_coarsest_res False \
    --per_pixel_bin_prediction False \
    --use_text_adapter True \
    --variance_focus 0.85 \
    --pixelshuffle_decoder True \
    --save_last_model