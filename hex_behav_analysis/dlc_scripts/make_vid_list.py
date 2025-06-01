from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from pathlib import Path
import sys

    
def get_video_paths(cohort_info, mode):
    """
    Extract video paths based on the specified mode and return counts.
    
    Args:
        cohort_info (dict): Dictionary containing cohort information with mice and sessions
        mode (str): Processing mode - either "analyse" or "label"
        
    Returns:
        tuple: (list of video paths, total videos considered, videos needing processing)
    """
    videos_to_process = []
    total_videos_considered = 0

    if mode == "analyse":
        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                
                # Count all videos where raw data is present
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    total_videos_considered += 1
                    
                    # Only add to processing list if coords_csv is missing
                    if cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos_to_process.append(video_directory)
                        print(video_directory)
                        
        return videos_to_process, total_videos_considered, len(videos_to_process)
    
    elif mode == "label":
        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                
                # Count all videos where raw data is present
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    total_videos_considered += 1
                    
                    # Add to processing list if coords_csv is missing
                    if cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos_to_process.append(video_directory)
                        
        return videos_to_process, total_videos_considered, len(videos_to_process)


if __name__ == "__main__":

    manual_input_cohort_folder = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE/Experiment")
    manual_input_mode = "analyse"  # Default mode, can be changed if needed

    # Handle command line arguments or use defaults
    if len(sys.argv) > 1:
        # Use command line argument for cohort directory
        cohort_directory = Path(sys.argv[1])
        print(f"Using cohort directory from command line: {cohort_directory}")
    else:
        # Use default/manual directory
        cohort_directory = manual_input_cohort_folder
        print(f"Using default cohort directory: {cohort_directory}")
    
    # Handle mode argument or use default
    if len(sys.argv) > 2:
        # Use command line argument for mode
        mode = sys.argv[2].lower()
        if mode not in ["analyse", "label"]:
            print(f"Error: Invalid mode '{mode}'. Must be 'analyse' or 'label'")
            sys.exit(1)
        print(f"Using mode from command line: {mode}")
    else:
        # Use default/manual mode
        mode = manual_input_mode
        print(f"Using default mode: {mode}")
    
    # Verify the directory exists
    if not cohort_directory.exists():
        print(f"Error: Cohort directory does not exist: {cohort_directory}")
        sys.exit(1)
    
    print(f"Processing cohort directory: {cohort_directory}")
    print(f"Mode: {mode}")
    
    print("Default Cohort_folder params: multi=True, OEAB_legacy=False")
    print("Check these are correct for your cohort.")
    try:
        cohort_info = Cohort_folder(cohort_directory, multi=True, OEAB_legacy=False).cohort
    except Exception as e:
        print(f"Error loading cohort information: {e}")
        sys.exit(1)
    
    # Get the video paths with counts
    videos, total_considered, videos_to_process = get_video_paths(cohort_info, mode)
    
    # Print summary statistics
    print(f"Total videos considered for {mode}: {total_considered}")
    print(f"Videos needing {mode}: {videos_to_process}")
    print(f"Videos already processed: {total_considered - videos_to_process}")

    # Save video paths to text file in cohort folder (filename based on mode)
    if mode == "analyse":
        output_file = cohort_directory / "videos_to_analyse.txt"
    elif mode == "label":
        output_file = cohort_directory / "videos_to_label.txt"
    
    try:
        with open(output_file, "w") as file:
            for video in videos:
                file.write(str(video) + "\n")
        
        print(f"Video paths saved to: {output_file}")
        
        if videos_to_process == 0:
            print(f"No videos need {mode} - all are already processed!")
        else:
            print(f"Ready to {mode} {videos_to_process} videos")
            
    except Exception as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)