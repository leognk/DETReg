#!/bin/bash
#SBATCH --job-name=DETReg_fs
###########RESOURCES###########
#SBATCH --nodelist="yagi33"
#SBATCH --partition=48-2
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=8
###############################
#SBATCH --output=log_fs.out
#SBATCH --error=log_fs.err
#SBATCH -v
###############################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
###############################
DATA_ROOT=/po2/goncalves/data
EXP_ROOT=/po2/goncalves/exps
EXP_DIR=${EXP_ROOT}/DETReg_fs
srun python -u main.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${EXP_DIR} \
    --dataset coco \
    --pretrain ${EXP_ROOT}/DETReg_fine_tune_base_classes_original/checkpoint.pth \
    --eval_every 100 \
    \
    #--resume ${EXP_DIR}/checkpoint.pth \
    #--eval