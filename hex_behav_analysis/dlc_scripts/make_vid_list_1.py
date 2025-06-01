from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from pathlib import Path

    
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

    cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE/Experiment")
    
    cohort_info = Cohort_folder(cohort_directory, multi=True, OEAB_legacy=False).cohort
    
    # Get the video paths with counts
    videos, total_considered, videos_to_analyse = get_video_paths(cohort_info, "analyse")
    
    # Print summary statistics
    print(f"Total videos considered for analysis: {total_considered}")
    print(f"Videos needing analysis: {videos_to_analyse}")
    print(f"Videos already processed: {total_considered - videos_to_analyse}")

    # Save video paths to text file in cohort folder
    output_file = cohort_directory / "videos_to_analyse.txt"
    with open(output_file, "w") as file:
        for video in videos:
            file.write(str(video) + "\n")
    
    print(f"Video paths saved to: {output_file}")