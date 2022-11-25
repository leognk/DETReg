#!/bin/bash
#SBATCH --job-name=Repr
###########RESOURCES###########
#SBATCH --nodelist="yagi33"
#SBATCH --partition=48-2
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
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