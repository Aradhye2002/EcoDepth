
export CUDA_VISIBLE_DEVICES=6

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nproc_per_node=1 \
    test.py \
    --dataset kitti --data_path ./data/kitti \
    --num_filters 32 32 32 --deconv_kernels 2 2 2\
    --flip_test  --ckpt_dir $1 \
    --vit_model google/vit-base-patch16-224 \
    --kitti_crop garg_crop \
    --do_kb_crop True \
    --kitti_split_to_half True \
    --save_visualize --save_eval_pngs \
    --exp_name testing_kitti \
    --max_depth 80.0 --max_depth_eval 80.0 --min_depth_eval 1e-3 \
    --median_scaling False \
    --batch_size 1
    ${@:2} \
