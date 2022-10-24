DATA_ROOT=/home/user/fiftyone
EXP_ROOT=exps
EXP_DIR=${EXP_ROOT}/DETReg_fs
python -u main.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${EXP_DIR} \
    --dataset coco \
    --pretrain ${EXP_ROOT}/DETReg_fine_tune_base_classes_original/checkpoint.pth \
    --eval_every 100 \
    \
    --resume ${EXP_DIR}/checkpoint.pth \
    --eval