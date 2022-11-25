#!/bin/bash
#SBATCH --job-name=Repr
###########RESOURCES###########
#SBATCH --nodelist="yagi32"
#SBATCH --partition=48-3
#SBATCH --gres=gpu:3
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
srun python -u compute_img_repr_points.py --dir /po2/goncalves/representer_points/arrays