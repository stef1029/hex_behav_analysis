import sys
from pathlib import Path

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder

def get_video_paths(cohort_info, mode):
    videos = []
    for mouse in cohort_info["mice"]:
        for session in cohort_info["mice"][mouse]["sessions"]:
            print(session)
            video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
            if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"]:
                if cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                    videos.append(video_directory)
                    print(video_directory)

    return videos

def main(cohort_directory, mode):
    cohort_info = Cohort_folder(cohort_directory, multi=True, OEAB_legacy = False).cohort
    # Get the video paths
    videos = get_video_paths(cohort_info, mode)
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
    if len(sys.argv) != 3:
        print("Usage: python make_vid_list.py <cohort_directory> <mode>")
        print("mode: 'analyse' or 'label'")
        sys.exit(1)

    cohort_directory = Path(sys.argv[1])
    mode = sys.argv[2]
    if mode not in ["analyse", "label"]:
        print("Invalid mode. Use 'analyse' or 'label'.")
        sys.exit(1)

    """
    Example usage: `python make_vid_list.py /path/to/cohort analyse`
    """

    main(cohort_directory, mode)
