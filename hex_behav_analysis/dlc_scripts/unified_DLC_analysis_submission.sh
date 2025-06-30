#!/bin/bash -l

#SBATCH --job-name=DLC_unified_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# =============================================================================
# CONFIGURATION - Update this path for different computers/setups
# =============================================================================
SCRIPTS_BASE_DIR="/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/dlc_scripts"

# =============================================================================
# Unified DeepLabCut Analysis Pipeline
# =============================================================================
# Usage: sbatch unified_dlc_pipeline.sh COHORT_FOLDER [NUM_GPUS] [MODE]
# Example: sbatch unified_dlc_pipeline.sh "/path/to/cohort1" 8 analyse
# =============================================================================

# Check if cohort folder is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide cohort folder path"
    echo "Usage: sbatch $0 COHORT_FOLDER [NUM_GPUS] [MODE]"
    echo "Example: sbatch $0 /cephfs2/dwelch/Behaviour/cohort1 8 analyse"
    exit 1
fi

COHORT_FOLDER="$1"
NUM_GPUS="${2:-8}"  # Default to 8 GPUs if not specified
MODE="${3:-analyse}"  # Default to "analyse" mode if not specified

echo "=== DeepLabCut Unified Pipeline ==="
echo "Starting at: $(date)"
echo "Cohort folder: $COHORT_FOLDER"
echo "Number of GPUs: $NUM_GPUS"
echo "Mode: $MODE"
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
NUM_LINES=$(wc -l < "$VIDEO_LIST_FILE")
echo "Found $NUM_LINES videos to $MODE"

if [ $NUM_LINES -eq 0 ]; then
    echo "No videos to $MODE. Exiting."
    exit 0
fi

# Step 3: Submit array job
echo ""
echo "=== Step 2: Submitting $MODE jobs ==="
echo "Submitting array job for $NUM_LINES videos using $NUM_GPUS GPUs"

# Note: This assumes your analysis script can handle both analyse and label modes
# You may need to modify newSH.sh or create separate scripts for different modes
ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis_DLC3.sh"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

# Set environment variable so the analysis script knows which file to read
export DLC_VIDEO_LIST_FILE="$(basename $VIDEO_LIST_FILE)"

JOB_SUBMISSION=$(sbatch --array=1-${NUM_LINES}%${NUM_GPUS} --chdir="$COHORT_FOLDER" --parsable "$ANALYSIS_SCRIPT")
ARRAY_JOB_ID=$(echo $JOB_SUBMISSION | cut -d';' -f1)

echo "Array job submitted with ID: $ARRAY_JOB_ID"
echo "Job range: 1-$NUM_LINES with max $NUM_GPUS concurrent jobs"
echo "Video list file: $VIDEO_LIST_FILE"

echo "Array job submitted with ID: $ARRAY_JOB_ID"
echo "Job range: 1-$NUM_LINES with max $NUM_GPUS concurrent jobs"
echo ""
echo "=== Monitoring Commands ==="
echo "Check job status: squeue -j $ARRAY_JOB_ID"
echo "Check detailed status: scontrol show job $ARRAY_JOB_ID"
echo "Cancel jobs: scancel $ARRAY_JOB_ID"
echo ""
echo "=== Log Files ==="
echo "Individual job logs will be created in: $COHORT_FOLDER"
echo "Pipeline log: $COHORT_FOLDER/$PIPELINE_LOG"
echo ""
echo "Pipeline setup completed at: $(date)"