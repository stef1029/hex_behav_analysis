#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=50G

# Hardcode test video path here
TEST_VIDEO_PATH="/path/to/your/test/video.mp4"

echo "=== DeepLabCut GPU Test ==="
echo "Starting job at $(date)"
echo "Test video: $TEST_VIDEO_PATH"

# Check available CUDA modules
echo "=== Available CUDA Modules ==="
module avail cuda

# Load CUDA module (try common versions for PyTorch 2.6.0)
echo "=== Loading CUDA Module ==="
module load cuda/12.1 || module load cuda/11.8 || module load cuda/12.4 || echo "No CUDA module loaded"

# Verify CUDA version
echo "=== CUDA Version ==="
nvcc --version

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Basic GPU check
echo "=== GPU Check ==="
nvidia-smi

# Activate environment
conda activate DEEPLABCUT3

# Quick CUDA availability check
echo "=== PyTorch CUDA Check ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Change to your DLC scripts directory
cd "/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Run your analysis script
echo "=== Running DeepLabCut Analysis ==="
python batch_analyse.py "$TEST_VIDEO_PATH" "0"

echo "Job finished at $(date)"