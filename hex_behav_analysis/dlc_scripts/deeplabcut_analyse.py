config = r'/cephfs2/srogers/New analysis pipeline/training_videos/DLC_Project_231212_193535_wtjx285-2a_raw_MP-SRC-2024-01-09/config.yaml'

# add perhaps config path as input function

import deeplabcut
import os


# get path movies
print("Analyzing videos")

videos_to_analyse = [r"/cephfs2/srogers/December training data/231220_115610_wtjx285-2d"]

# Train new videos (Check if previous training already has coordinates)
deeplabcut.analyze_videos(config = config,
                          videos = videos_to_analyse, 
                          videotype='.avi', 
                          save_as_csv=True, 
                          gputouse=int(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("Video analyzed")

# # create labeled data
# print("Creating labeled videos")
# deeplabcut.create_labeled_video(config = config, 
#                                 videos = videos_to_analyse, 
#                                 save_frames=False, 
#                                 draw_skeleton=False, 
#                                 fastmode = True)
# print("Labeled videos created")