NPROC_PER_NODE=8
NO_CPU_CORES=64

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$((NO_CPU_CORES / NPROC_PER_NODE))
export TOKENIZERS_PARALLELISM=false

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
    --batch_size 4 \
    --dataset nyudepthv2 \
    --data_path /home/aradhye/scratch/Datasets/ \
    --weight_decay 0.1 \
    --crop_h 480 \
    --crop_w 640 \
    --num_filters 32 32 32 \
    --deconv_kernels 2 2 2 \
    --flip_test \
    --layer_decay 0.9 \
    --drop_path_rate 0.3 \
    ${@:2} \
    --pro_bar True \
    --save_depths_gray \
    --save_depths_color \
    --vit_model google/vit-base-patch16-224 \
    --learning_rate_schedule one_cycle \
    --gradient_accumulation False \
    --num_of_diffusion_step 1 \
    --epochs 25 \
    --log_dir log_dir_vision04 \
    --log_in_wandb True \
    --eigen_crop_in_dataloader_itself_for_nyu False \
    --min_depth_eval 1e-3 \
    --max_depth_eval 10 \
    --exp_name training_nyu \
    --no_of_classes 100 \
    --variance_focus 0.85 \
    --cutflip=False \
    --save_last_model