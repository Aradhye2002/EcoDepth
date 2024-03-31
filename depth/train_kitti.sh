NPROC_PER_NODE=1 #8
NO_CPU_CORES=64

export CUDA_VISIBLE_DEVICES=4 #,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$((NO_CPU_CORES / NPROC_PER_NODE))
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nproc_per_node=${NPROC_PER_NODE} \
    train.py \
    --batch_size 1 \
    --dataset kitti \
    --data_path ./data/kitti \
    --weight_decay 0.1 \
    --num_filters 32 32 32 \
    --deconv_kernels 2 2 2 \
    --flip_test \
    --layer_decay 0.9 \
    --drop_path_rate 0.3 \
    --log_dir ./log_dir \
    ${@:2} \
    --exp_name kitti \
    --pro_bar True \
    --save_depths_gray \
    --save_depths_color \
    --vit_model google/vit-base-patch16-224 \
    --learning_rate_schedule one_cycle \
    --gradient_accumulation False \
    --epochs 25 \
    --log_in_wandb True \
    --min_depth_eval 1e-3 \
    --max_depth_eval 80 \
    --max_depth 80 \
    --kitti_crop garg_crop \
    --do_kb_crop True \
    --cutflip False \
    --use_right False \
    --finetune_on_another_dataset False \
    --variance_focus 0.85 \
    --save_last_model
