#!/bin/bash -l

#SBATCH --partition=agpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL

# Define the index of the video this particular job should process
VIDEO_INDEX=$SLURM_ARRAY_TASK_ID

# Path to your list of videos
VIDEO_LIST="videos_to_analyse.txt"

# Extract the specific video path for this job
VIDEO_PATH=$(sed -n "${VIDEO_INDEX}p" $VIDEO_LIST)

# Extract session ID (parent directory name) instead of video basename
SESSION_ID=$(basename "$(dirname "$VIDEO_PATH")")

echo "Processing video: $VIDEO_PATH"

# Get the current directory where the script is running from
CURRENT_DIR=$(pwd)

# Create a temporary log file using session ID
TEMP_LOG="${CURRENT_DIR}/${SESSION_ID}_temp.log"

# Redirect all output to the temporary log file
exec > "$TEMP_LOG" 2>&1

echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Video: $VIDEO_PATH"
echo "Session ID: $SESSION_ID"

# Load CUDA module
echo "=== Loading CUDA Module ==="
module load cuda/12.1 || module load cuda/11.8 || module load cuda/12.4 || echo "No CUDA module loaded"

# Show what SLURM has set for GPU visibility
echo "SLURM GPU allocation: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES (set by SLURM): $CUDA_VISIBLE_DEVICES"

# Don't override CUDA_VISIBLE_DEVICES - let SLURM handle it
# SLURM automatically sets this when you use --gres=gpu:1

# Activate environment
conda activate DEEPLABCUT3

# Enhanced CUDA availability check
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
import os
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count visible to PyTorch: {torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Change to your DLC scripts directory
cd "/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Run the Python script - pass GPU ID as 0 since SLURM maps the allocated GPU to index 0
echo "Running batch_analyse.py..."
python batch_analyse.py "$VIDEO_PATH" 0
PYTHON_EXIT_CODE=$?

# Check if the Python script failed and rename the log file accordingly
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $PYTHON_EXIT_CODE"
    echo "Job failed at $(date)"
    
    # Rename the temporary log file to include FAILED in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_FAILED.log"
else
    echo "Python script completed successfully"
    echo "Job completed at $(date)"
    
    # Rename the temporary log file to include SUCCESS in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_SUCCESS.log"
fi

# Close the file descriptors before moving the file
exec >&- 2>&-

# Move the temporary log to the final log with the appropriate status in the filename
mv "$TEMP_LOG" "$FINAL_LOG"

# Exit with the Python script's exit code
exit $PYTHON_EXIT_CODE