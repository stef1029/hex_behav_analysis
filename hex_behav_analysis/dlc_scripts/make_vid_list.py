import sys
from pathlib import Path

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder

def get_video_paths(cohort_info, mode, refresh=False):
    videos = []
    for mouse in cohort_info["mice"]:
        for session in cohort_info["mice"][mouse]["sessions"]:
            print(session)
            video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
            if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"]:
                # If refresh is True, add all available videos
                # If refresh is False, only add videos that haven't been processed by DLC
                if refresh or cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                    videos.append(video_directory)
                    print(video_directory)

    return videos

def main(cohort_directory, mode, refresh=False):
    cohort_info = Cohort_folder(cohort_directory, multi=True, OEAB_legacy = False).cohort
    # Get the video paths
    videos = get_video_paths(cohort_info, mode, refresh)
    print(f"Number of videos to {mode}: {len(videos)}")

    # Save as .txt file in cohort folder:
    filename = f"videos_to_{mode}.txt"
    with open(cohort_directory / filename, "w") as file:
        for video in videos:
            file.write(str(video) + "\n")

    # Create a .signal file to indicate completion
    signal_file = cohort_directory / "make_vid_list.signal"
    with open(signal_file, "w") as file:
        file.write("done")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python make_vid_list.py <cohort_directory> <mode> [refresh]")
        print("mode: 'analyse' or 'label'")
        print("refresh: optional flag, set to 'True' to include all videos regardless of DLC processing status")
        sys.exit(1)

    cohort_directory = Path(sys.argv[1])
    mode = sys.argv[2]
    if mode not in ["analyse", "label"]:
        print("Invalid mode. Use 'analyse' or 'label'.")
        sys.exit(1)
    
    # Check if refresh parameter was provided
    refresh = False
    if len(sys.argv) == 4:
        refresh_arg = sys.argv[3].lower()
        if refresh_arg == "true":
            refresh = True
        elif refresh_arg != "false":
            print("Invalid refresh value. Use 'True' or 'False'.")
            sys.exit(1)

    """
    Example usage: 
    - Standard: `python make_vid_list.py /path/to/cohort analyse`
    - With refresh: `python make_vid_list.py /path/to/cohort analyse True`
    """

    main(cohort_directory, mode, refresh)