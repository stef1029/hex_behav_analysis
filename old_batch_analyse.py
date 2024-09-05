import deeplabcut
import os
import time
from Cohort_folder import Cohort_folder
from pathlib import Path
import multiprocessing as mp

# goal of this script:
# use the pretrained model from deeplabcut to analyse either a list of videos provided by the cohort object or
# individual paths that I dictate.

# the config file is the directory to the model that i trained with the original 5 example videos
# ideally would also make use of the multiple gpus available in a multiprocessing type way

config = r'/cephfs2/srogers/New analysis pipeline/training_videos/DLC_Project_231212_193535_wtjx285-2a_raw_MP-SRC-2024-01-09/config.yaml'


def analyse(video_path, gpu_id):
    start_time = time.perf_counter()
    print(f"Analyzing {str(video_path)}")
    print(f"Using GPU {gpu_id}")
    deeplabcut.analyze_videos(config = config,
                          videos = [video_path], 
                          videotype='.avi', 
                          save_as_csv=True, 
                          gputouse=gpu_id)
    # give time in minutes and seconds, rounded:
    print(f"""Analysis of {str(video_path)} complete, took {round((time.perf_counter() - start_time)/60, 2)} minutes 
                \r and {round((time.perf_counter() - start_time)%60, 2)} seconds""")
    
def label_video(video_path):
    start_time = time.perf_counter()
    print(F"Creating labeled video for {str(video_path)}")
    deeplabcut.create_labeled_video(config = config, 
                                    videos = [video_path], 
                                    save_frames=False, 
                                    draw_skeleton=True, 
                                    fastmode = True)
    print(f"""Labeled videos created for {str(video_path)}, took {round((time.perf_counter() - start_time)/60, 2)} minutes
          \r and {round((time.perf_counter() - start_time)%60, 2)} seconds""")
    
def get_video_paths(cohort_info, mode):

    if type == "analyse":
        videos = []

        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    if not cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos.append(video_directory)
        return videos
    
    elif type == "label":
        videos = []

        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                video_directory = Path(cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])
                if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    if not cohort_info["mice"][mouse]["sessions"][session]["processed_data"]["DLC"]["coords_csv"] == "None":
                        videos.append(video_directory)
        return videos


def worker(gpu_id, queue):
    while not queue.empty():
        try:
            video_path = queue.get_nowait()
        except queue.Empty:
            break

        analyse(video_path, gpu_id)
        
def analyse_across_gpus(video_paths, num_gpus):
    # Create a queue and enqueue all video paths
    video_queue = mp.Queue()
    for video_path in video_paths:
        video_queue.put(video_path)
    
    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        print(f"Starting worker {gpu_id}")
        p = mp.Process(target=worker, args=(gpu_id, video_queue))
        p.start()
        workers.append(p)
    
    # Wait for all workers to finish
    for p in workers:
        p.join()

def label_videos_MP(video_paths):
    
    max_processes = mp.cpu_count()

    with mp.Pool(processes = max_processes) as pool:
        pool.map(label_video, video_paths)



if __name__ == '__main__':

    cohort_folder = Path(r"/cephfs2/srogers/December training data")

    cohort_info = Cohort_folder(cohort_folder).cohort

    videos_to_analyse = get_video_paths(cohort_info, "analyse")
    print(videos_to_analyse)

    # videos_to_analyse = [r"/cephfs2/srogers/December training data/231220_115610_wtjx285-2d"]

    # gpus_avaliable = [int(gpu) for gpu in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]

    # analyse_across_gpus(videos_to_analyse, len(gpus_avaliable))

    # videos_to_label = get_video_paths(cohort_info, "label")

    # label_videos_MP(videos_to_label)