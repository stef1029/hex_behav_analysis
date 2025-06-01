#!/bin/bash -l

#SBATCH --partition=agpu
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
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

# Extract session ID (parent directory name) instead of video basename
SESSION_ID=$(basename "$(dirname "$VIDEO_PATH")")

echo "Processing video: $VIDEO_PATH on GPU: $GPU_ID"

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
echo "GPU ID: $GPU_ID"

# Load CUDA module
echo "=== Loading CUDA Module ==="
module load cuda/12.1 || module load cuda/11.8 || module load cuda/12.4 || echo "No CUDA module loaded"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate environment
conda activate DEEPLABCUT3

# Quick CUDA availability check
echo "=== PyTorch CUDA Check ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Change to your DLC scripts directory
cd "/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Run the Python script and capture its exit code
echo "Running batch_analyse.py..."
python batch_analyse.py "$VIDEO_PATH" "$GPU_ID" 
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