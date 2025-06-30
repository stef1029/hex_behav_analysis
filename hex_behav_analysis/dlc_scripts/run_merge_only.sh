#!/bin/bash -l

#SBATCH --job-name=DLC_merge_only
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# =============================================================================
# Run DeepLabCut Merge Only
# =============================================================================
# This script runs only the merge step for existing split files
# Usage: sbatch run_merge_only.sh COHORT_FOLDER [SPLITS_PER_VIDEO]
# =============================================================================

# Check if cohort folder is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide cohort folder path"
    echo "Usage: sbatch $0 COHORT_FOLDER [SPLITS_PER_VIDEO]"
    echo "Example: sbatch $0 /cephfs2/dwelch/Behaviour/cohort1 8"
    exit 1
fi

COHORT_FOLDER="$1"
SPLITS_PER_VIDEO="${2:-8}"  # Default to 8 splits if not specified

echo "=== DeepLabCut Merge Only ==="
echo "Starting at: $(date)"
echo "Cohort folder: $COHORT_FOLDER"
echo "Splits per video: $SPLITS_PER_VIDEO"
echo "Running on node: $(hostname)"

# Check if cohort folder exists
if [ ! -d "$COHORT_FOLDER" ]; then
    echo "Error: Cohort folder does not exist: $COHORT_FOLDER"
    exit 1
fi

# Configuration
SCRIPTS_BASE_DIR="/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# Check for existing video list file
VIDEO_LIST_FILE="$COHORT_FOLDER/videos_to_analyse.txt"

if [ ! -f "$VIDEO_LIST_FILE" ]; then
    echo "Video list file not found at: $VIDEO_LIST_FILE"
    echo "Generating new video list..."
    
    # Activate conda environment
    conda activate behaviour_analysis
    
    # Generate video list
    python "$SCRIPTS_BASE_DIR/make_vid_list.py" "$COHORT_FOLDER" "analyse"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate video list"
        exit 1
    fi
    
    if [ ! -f "$VIDEO_LIST_FILE" ]; then
        echo "Error: Video list still not found after generation"
        exit 1
    fi
fi

# Count videos
NUM_VIDEOS=$(wc -l < "$VIDEO_LIST_FILE")
echo "Found $NUM_VIDEOS videos to process"

if [ $NUM_VIDEOS -eq 0 ]; then
    echo "No videos found to merge. Exiting."
    exit 0
fi

# Change to cohort directory
cd "$COHORT_FOLDER"

# Activate the required environment
echo "Activating conda environment: behaviour_analysis"
conda activate behaviour_analysis

# Run the merge Python script
echo "Running merge script..."
python "$SCRIPTS_BASE_DIR/merge_dlc_splits.py" "$VIDEO_LIST_FILE" "$SPLITS_PER_VIDEO"

MERGE_EXIT_CODE=$?

if [ $MERGE_EXIT_CODE -eq 0 ]; then
    echo "Merge completed successfully at: $(date)"
else
    echo "Merge script encountered issues at: $(date)"
    echo "Exit code: $MERGE_EXIT_CODE"
    echo "Check the output above for details about which videos were processed successfully"
fi

echo "=== Summary ==="
echo "Total videos in list: $NUM_VIDEOS"
echo "Check the merge script output above to see which videos were successfully merged"
echo "Videos with missing splits were skipped and can be rerun individually"

exit $MERGE_EXIT_CODE