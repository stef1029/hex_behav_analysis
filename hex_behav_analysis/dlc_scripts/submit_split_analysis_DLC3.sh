#!/bin/bash -l

#SBATCH --partition=agpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --mail-type=FAIL

# Define the index for this job
JOB_INDEX=$SLURM_ARRAY_TASK_ID

# Path to the split information file
SPLIT_INFO_FILE="${DLC_SPLIT_INFO_FILE:-video_splits_info.txt}"
SPLITS_PER_VIDEO="${DLC_SPLITS_PER_VIDEO:-8}"

# Extract the specific split information for this job
SPLIT_INFO=$(sed -n "${JOB_INDEX}p" $SPLIT_INFO_FILE)

# Parse the split information
VIDEO_INDEX=$(echo $SPLIT_INFO | cut -d':' -f1)
SPLIT_INDEX=$(echo $SPLIT_INFO | cut -d':' -f2)
VIDEO_PATH=$(echo $SPLIT_INFO | cut -d':' -f3-)

# Extract session ID (parent directory name)
SESSION_ID=$(basename "$(dirname "$VIDEO_PATH")")

echo "Processing video split: $VIDEO_PATH (Video $VIDEO_INDEX, Split $SPLIT_INDEX/$SPLITS_PER_VIDEO)"

# Get the current directory where the script is running from
CURRENT_DIR=$(pwd)

# Create a temporary log file using session ID and split info
TEMP_LOG="${CURRENT_DIR}/${SESSION_ID}_split${SPLIT_INDEX}_temp.log"

# Redirect all output to the temporary log file
exec > "$TEMP_LOG" 2>&1

echo "Starting split job at $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Video: $VIDEO_PATH"
echo "Session ID: $SESSION_ID"
echo "Split: $SPLIT_INDEX of $SPLITS_PER_VIDEO"
echo "memory in use is 120gb"

# Load CUDA module and capture which version was loaded
echo "=== Loading CUDA Module ==="
if module load cuda/12.4 2>/dev/null; then
    echo "Successfully loaded CUDA 12.4"
    CUDA_MODULE_VERSION="12.4"
elif module load cuda/11.8 2>/dev/null; then
    echo "Successfully loaded CUDA 11.8"
    CUDA_MODULE_VERSION="11.8"
elif module load cuda/12.1 2>/dev/null; then
    echo "Successfully loaded CUDA 12.1"
    CUDA_MODULE_VERSION="12.1"
else
    echo "WARNING: No CUDA module could be loaded"
    CUDA_MODULE_VERSION="None"
fi

# Print system CUDA version if available
echo "=== System CUDA Version Check ==="
if command -v nvcc &> /dev/null; then
    echo "NVCC version:"
    nvcc --version | grep "release" || echo "Could not parse NVCC version"
else
    echo "NVCC not found in PATH"
fi

# Show what SLURM has set for GPU visibility
echo "=== SLURM GPU Configuration ==="
echo "SLURM GPU allocation: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES (set by SLURM): $CUDA_VISIBLE_DEVICES"
echo "Loaded CUDA module version: $CUDA_MODULE_VERSION"

# Activate environment
conda activate DEEPLABCUT3

# Enhanced CUDA and PyTorch version check
echo "=== PyTorch and CUDA Compatibility Check ==="
python -c "
import torch
import os
import sys

print('=== PyTorch Version Information ===')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')

print('\n=== CUDA Availability ===')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count visible to PyTorch: {torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')

if torch.cuda.is_available():
    print('\n=== GPU Information ===')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  - Compute capability: {props.major}.{props.minor}')
        print(f'  - Total memory: {props.total_memory / 1024**3:.2f} GB')
        
    # Test CUDA functionality
    try:
        test_tensor = torch.cuda.FloatTensor([1.0, 2.0, 3.0])
        print('\n✓ CUDA tensor creation test passed')
    except Exception as e:
        print(f'\n✗ CUDA tensor creation test failed: {e}')
else:
    print('\n✗ CUDA is not available to PyTorch')
    sys.exit(1)
"

# Check if the Python CUDA check failed
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA availability check failed"
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_split${SPLIT_INDEX}_CUDA_FAILED.log"
    exec >&- 2>&-
    mv "$TEMP_LOG" "$FINAL_LOG"
    exit 1
fi

# Change to your DLC scripts directory
cd "/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Run the split analysis Python script
echo "Running batch_analyse_split.py..."
python batch_analyse_split.py "$VIDEO_PATH" 0 "$SPLIT_INDEX" "$SPLITS_PER_VIDEO"
PYTHON_EXIT_CODE=$?

# Check if the Python script failed and rename the log file accordingly
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $PYTHON_EXIT_CODE"
    echo "Job failed at $(date)"
    
    # Rename the temporary log file to include FAILED in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_split${SPLIT_INDEX}_FAILED.log"
else
    echo "Python script completed successfully"
    echo "Job completed at $(date)"
    
    # Rename the temporary log file to include SUCCESS in the name
    FINAL_LOG="${CURRENT_DIR}/${SESSION_ID}_split${SPLIT_INDEX}_SUCCESS.log"
fi

# Close the file descriptors before moving the file
exec >&- 2>&-

# Move the temporary log to the final log with the appropriate status in the filename
mv "$TEMP_LOG" "$FINAL_LOG"

# Exit with the Python script's exit code
exit $PYTHON_EXIT_CODE