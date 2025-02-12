from deeplabcut import analyze_videos
import time
import sys



# config = r'/cephfs2/srogers/New_analysis_pipeline/training_videos/DLC_Project_231212_193535_wtjx285-2a_raw_MP-SRC-2024-01-09/config.yaml'
config = r'/cephfs2/dwelch/6-choice_behaviour_DLC_model/config.yaml'

def analyse(video_path, gpu_id):
    start_time = time.perf_counter()
    print(f"Analyzing {str(video_path)}")
    print(f"Using GPU {gpu_id}")
    analyze_videos(config = config,
                          videos = [video_path], 
                          videotype='.avi', 
                          save_as_csv=True, 
                          gputouse=gpu_id)
    # give time in minutes and seconds, rounded:
    print(f"""Analysis of {str(video_path)} complete. 
                \rTook: {round((time.perf_counter() - start_time)/60, 2)} minutes 
                \rand {round((time.perf_counter() - start_time)%60, 2)} seconds""")
    

if __name__ == "__main__":
    video_path = sys.argv[1]
    gpu_id = int(sys.argv[2])

    analyse(video_path, gpu_id)
