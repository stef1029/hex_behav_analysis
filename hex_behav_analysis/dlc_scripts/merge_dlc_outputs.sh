#!/bin/bash -l

#SBATCH --job-name=DLC_merge
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# =============================================================================
# Merge DeepLabCut Split Outputs
# =============================================================================
# This script merges the split DLC outputs back into single files per video
# =============================================================================

VIDEO_LIST_FILE="$1"
SPLITS_PER_VIDEO="$2"

echo "=== DeepLabCut Merge Job ==="
echo "Starting at: $(date)"
echo "Video list: $VIDEO_LIST_FILE"
echo "Splits per video: $SPLITS_PER_VIDEO"

# Activate the required environment
conda activate behaviour_analysis

# Get the script directory
SCRIPTS_BASE_DIR="/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Run the merge Python script
python "$SCRIPTS_BASE_DIR/merge_dlc_splits.py" "$VIDEO_LIST_FILE" "$SPLITS_PER_VIDEO"

if [ $? -eq 0 ]; then
    echo "Merge completed successfully at: $(date)"
else
    echo "Merge failed at: $(date)"
    exit 1
fi