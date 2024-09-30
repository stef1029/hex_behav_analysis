import deeplabcut

# Path to the DeepLabCut config file
config_path = r"C:\Data\temp_data\DLC_Project_231212_193535_wtjx285-2a_raw_MP-SRC-2024-01-09\config.yaml"

# Specify the path to the video file you analyzed
video_paths = [r"C:\Data\temp_data\240917_153243_mtao89-1e\240917_153243_mtao89-1e_raw_MP.avi"]

destfolder = r'C:\Data\temp_data\240917_153243_mtao89-1e'

# Create labeled video
deeplabcut.create_labeled_video(
    config_path,
    video_paths,
    destfolder=destfolder,
    draw_skeleton=True,
    save_frames=False
)