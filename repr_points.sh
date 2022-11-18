#!/bin/bash
#SBATCH --job-name=Repr
###########RESOURCES###########
#SBATCH --nodelist="yagi32"
#SBATCH --partition=48-3
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
###############################
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH -v
###############################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
###############################
DATA_ROOT=/po2/goncalves/data
EXP_ROOT=/po2/goncalves/exps
REPR_DIR=/po2/goncalves/representer_points
srun python -u precompute_repr_points.py \
    --data_root ${DATA_ROOT} \
    --dataset coco \
    --pretrain ${EXP_ROOT}/DETReg_fs/checkpoint.pth \
    --num_workers 4 \
    --batch_size 4 \
    --repr_dir ${REPR_DIR}