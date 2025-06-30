#!/bin/bash -l

#SBATCH --job-name=DLC_split_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# =============================================================================
# CONFIGURATION - Update this path for different computers/setups
# =============================================================================
SCRIPTS_BASE_DIR="/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# =============================================================================
# Split Video DeepLabCut Analysis Pipeline
# =============================================================================
# Usage: sbatch split_unified_dlc_pipeline.sh COHORT_FOLDER [MODE]
# Example: sbatch split_unified_dlc_pipeline.sh "/path/to/cohort1" analyse
# =============================================================================

# Check if cohort folder is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide cohort folder path"
    echo "Usage: sbatch $0 COHORT_FOLDER [MODE]"
    echo "Example: sbatch $0 /cephfs2/dwelch/Behaviour/cohort1 analyse"
    exit 1
fi

COHORT_FOLDER="$1"
MODE="${2:-analyse}"  # Default to "analyse" mode if not specified
SPLITS_PER_VIDEO=8  # Number of splits per video

echo "=== DeepLabCut Split Video Pipeline ==="
echo "Starting at: $(date)"
echo "Cohort folder: $COHORT_FOLDER"
echo "Mode: $MODE"
echo "Splits per video: $SPLITS_PER_VIDEO"
echo "Running on node: $(hostname)"

# Validate mode
if [[ "$MODE" != "analyse" && "$MODE" != "label" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'analyse' or 'label'"
    exit 1
fi

# Check if cohort folder exists
if [ ! -d "$COHORT_FOLDER" ]; then
    echo "Error: Cohort folder does not exist: $COHORT_FOLDER"
    exit 1
fi

# Step 1: Generate video list
echo ""
echo "=== Step 1: Generating video list ==="

# Activate the required conda environment
echo "Activating conda environment: behaviour_analysis"
conda activate behaviour_analysis

# Use the configured script directory
SCRIPT_DIR="$SCRIPTS_BASE_DIR"
echo "Script directory: $SCRIPT_DIR"

# Check if make_vid_list.py exists
if [ ! -f "$SCRIPT_DIR/make_vid_list.py" ]; then
    echo "Error: make_vid_list.py not found in $SCRIPT_DIR"
    echo "Please update SCRIPTS_BASE_DIR variable at the top of this script"
    echo "Contents of script directory:"
    ls -la "$SCRIPT_DIR" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

# Run make_vid_list.py with cohort folder and mode
python "$SCRIPT_DIR/make_vid_list.py" "$COHORT_FOLDER" "$MODE"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate video list"
    exit 1
fi

# Check for the correct output file based on mode (use full path)
if [ "$MODE" == "analyse" ]; then
    VIDEO_LIST_FILE="$COHORT_FOLDER/videos_to_analyse.txt"
elif [ "$MODE" == "label" ]; then
    VIDEO_LIST_FILE="$COHORT_FOLDER/videos_to_label.txt"
fi

# Check if the video list file was created
if [ ! -f "$VIDEO_LIST_FILE" ]; then
    echo "Error: $(basename $VIDEO_LIST_FILE) not found at $VIDEO_LIST_FILE"
    echo "Contents of cohort directory:"
    ls -la "$COHORT_FOLDER/"
    exit 1
fi

# Step 2: Count videos
NUM_VIDEOS=$(wc -l < "$VIDEO_LIST_FILE")
echo "Found $NUM_VIDEOS videos to $MODE"

if [ $NUM_VIDEOS -eq 0 ]; then
    echo "No videos to $MODE. Exiting."
    exit 0
fi

# Step 3: Create split information file
echo ""
echo "=== Step 2: Creating split information ==="
SPLIT_INFO_FILE="$COHORT_FOLDER/video_splits_info.txt"

# Clear any existing split info file
> "$SPLIT_INFO_FILE"

# Generate split information for each video
VIDEO_INDEX=1
while IFS= read -r VIDEO_PATH; do
    for SPLIT_INDEX in $(seq 1 $SPLITS_PER_VIDEO); do
        echo "${VIDEO_INDEX}:${SPLIT_INDEX}:${VIDEO_PATH}" >> "$SPLIT_INFO_FILE"
    done
    ((VIDEO_INDEX++))
done < "$VIDEO_LIST_FILE"

TOTAL_JOBS=$(wc -l < "$SPLIT_INFO_FILE")
echo "Created split information for $TOTAL_JOBS jobs ($NUM_VIDEOS videos × $SPLITS_PER_VIDEO splits)"

# Step 4: Submit array job for split processing
echo ""
echo "=== Step 3: Submitting split $MODE jobs ==="
echo "Submitting array job for $TOTAL_JOBS split jobs"

# Use the split analysis script
SPLIT_ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_split_analysis_DLC3.sh"

# Check if the split analysis script exists, if not we'll create it
if [ ! -f "$SPLIT_ANALYSIS_SCRIPT" ]; then
    echo "Split analysis script not found. Please ensure submit_split_analysis_DLC3.sh exists in $SCRIPT_DIR"
    exit 1
fi

# Set environment variable so the analysis script knows which file to read
export DLC_SPLIT_INFO_FILE="video_splits_info.txt"
export DLC_SPLITS_PER_VIDEO="$SPLITS_PER_VIDEO"

# Submit the array job with all splits
JOB_SUBMISSION=$(sbatch --array=1-${TOTAL_JOBS} --chdir="$COHORT_FOLDER" --parsable "$SPLIT_ANALYSIS_SCRIPT")
ARRAY_JOB_ID=$(echo $JOB_SUBMISSION | cut -d';' -f1)

echo "Array job submitted with ID: $ARRAY_JOB_ID"
echo "Job range: 1-$TOTAL_JOBS"
echo "Split info file: $SPLIT_INFO_FILE"

# Step 5: Submit merge job that depends on the array job completion
echo ""
echo "=== Step 4: Submitting merge job ==="

MERGE_SCRIPT="$SCRIPT_DIR/merge_dlc_outputs.sh"

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "Merge script not found. Please ensure merge_dlc_outputs.sh exists in $SCRIPT_DIR"
    exit 1
fi

# Submit merge job with dependency on the array job
MERGE_JOB_SUBMISSION=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} --chdir="$COHORT_FOLDER" --parsable "$MERGE_SCRIPT" "$VIDEO_LIST_FILE" "$SPLITS_PER_VIDEO")
MERGE_JOB_ID=$(echo $MERGE_JOB_SUBMISSION | cut -d';' -f1)

echo "Merge job submitted with ID: $MERGE_JOB_ID (depends on array job $ARRAY_JOB_ID)"

echo ""
echo "=== Monitoring Commands ==="
echo "Check split job status: squeue -j $ARRAY_JOB_ID"
echo "Check merge job status: squeue -j $MERGE_JOB_ID"
echo "Check detailed status: scontrol show job $ARRAY_JOB_ID"
echo "Cancel all jobs: scancel $ARRAY_JOB_ID $MERGE_JOB_ID"
echo ""
echo "=== Log Files ==="
echo "Individual split job logs will be created in: $COHORT_FOLDER"
echo "Merge job log will be: $COHORT_FOLDER/slurm-${MERGE_JOB_ID}.out"
echo ""
echo "Pipeline setup completed at: $(date)"