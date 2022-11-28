DATA_ROOT=/home/user/fiftyone
EXP_ROOT=exps
REPR_DIR=representer_points
python -u precompute_repr_points.py \
    --data_root ${DATA_ROOT} \
    --dataset coco \
    --pretrain ${EXP_ROOT}/DETReg_fs/checkpoint.pth \
    --num_workers 4 \
    --batch_size 4 \
    --repr_dir ${REPR_DIR}