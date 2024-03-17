NPROC_PER_NODE=1
NO_CPU_CORES=64

export CUDA_VISIBLE_DEVICES=0

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
    test.py \
    --dataset kitti --data_path /home/suraj/scratch/Datasets/kitti_depth_from_ash \
    --num_filters 32 32 32 --deconv_kernels 2 2 2\
    --flip_test  --ckpt_dir /home/suraj/scratch/depth/MDP_CVPR/depth/our_best_models/kitti_best/best_abs_rel_0.0485.ckpt \
    --crop_h 480 --crop_w 480 \
    --vit_model google/vit-base-patch16-224 \
    --latent_scaling \
    --num_of_diffusion_step 1 \
    --kitti_crop garg_crop \
    --do_kb_crop True \
    --kitti_split_to_half  \
    --exp_name testing_kitti \
    --max_depth 80.0 --max_depth_eval 80.0 --min_depth_eval 1e-3 \
    --median_scaling False \
    --batch_size 1
    ${@:2} \
