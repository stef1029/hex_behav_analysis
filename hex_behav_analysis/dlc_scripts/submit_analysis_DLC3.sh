#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
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

# Extract just the filename without extension from the path for use in output naming
VIDEO_BASENAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

echo "Processing video: $VIDEO_PATH on GPU: $GPU_ID"

# Get the current directory where the script is running from
CURRENT_DIR=$(pwd)

# Create a temporary log file
TEMP_LOG="${CURRENT_DIR}/${VIDEO_BASENAME}_temp.log"

# Redirect all output to the temporary log file
exec > "$TEMP_LOG" 2>&1

echo "Starting job at $(date)"
echo "Video: $VIDEO_PATH"
echo "GPU ID: $GPU_ID"

conda activate DEEPLABCUT3

# cd "/cephfs2/srogers/New_analysis_pipeline/Scripts"
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
    FINAL_LOG="${CURRENT_DIR}/${VIDEO_BASENAME}_FAILED.log"
else
    echo "Python script completed successfully"
    echo "Job completed at $(date)"
    
    # Rename the temporary log file to include SUCCESS in the name
    FINAL_LOG="${CURRENT_DIR}/${VIDEO_BASENAME}_SUCCESS.log"
fi

# Close the file descriptors before moving the file
exec >&- 2>&-

# Move the temporary log to the final log with the appropriate status in the filename
mv "$TEMP_LOG" "$FINAL_LOG"

# Exit with the Python script's exit code
exit $PYTHON_EXIT_CODE