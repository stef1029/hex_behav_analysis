#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL

# Calculate the GPU ID to use by this task
NUM_GPUS=1
GPU_ID=$((SLURM_ARRAY_TASK_ID % NUM_GPUS))

# Define the index of the video this particular job should process
VIDEO_INDEX=$SLURM_ARRAY_TASK_ID

# Path to your list of videos
VIDEO_LIST="videos_to_analyse.txt"

# Extract the specific video path for this job
VIDEO_PATH=$(sed -n "${VIDEO_INDEX}p" $VIDEO_LIST)

echo "Processing video: $VIDEO_PATH on GPU: $GPU_ID"

conda activate DEEPLABCUT3

# cd "/cephfs2/srogers/New_analysis_pipeline/Scripts"
cd "/lmb/home/srogers/dev/projects/July_cohort_scripts/hex_behav_analysis/dlc_scripts"

# Ensure your Python script can take a video path and a GPU ID as arguments
python batch_analyse.py "$VIDEO_PATH" "$GPU_ID"