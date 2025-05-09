from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from pathlib import Path

    
def get_video_paths(cohort_info, mode):

    if mode == "analyse":
        videos = []

        for mouse in cohort_info["mice"]:
            # print(mouse)
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                # print(video_directory)
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    if cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos.append(video_directory)
                        print(video_directory)
        return videos
    
    elif mode == "label":
        videos = []

        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    if cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos.append(video_directory)
        return videos

if __name__ == "__main__":

    cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE")
    
    cohort_info = Cohort_folder(cohort_directory, multi = True, OEAB_legacy = False).cohort
    # print(cohort_info)
    # Get the video paths
    videos = get_video_paths(cohort_info, "analyse")
    print(f"Number of videos to analyse: {len(videos)}")

    # save as .txt file in cohort folder:
    with open(cohort_directory / "videos_to_analyse.txt", "w") as file:
        for video in videos:
            file.write(str(video) + "\n")

    
    