#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL

nvidia-smi -L
module load cuda/10.1

conda activate DEEPLABCUT

cd "/cephfs2/srogers/New analysis pipeline/Scripts"

python "deeplabcut_analyse.py"
