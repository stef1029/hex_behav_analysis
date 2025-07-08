"""
DeepLabCut video analysis script with malloc_trim fix for memory issues.
"""

from deeplabcut import analyze_videos
import time
import sys
import gc
import threading
import ctypes

# Configuration
video_path = "/cephfs2/srogers/Behaviour/test/250522_114207_mtao106-3a_video.avi"
gpu_id = 0
config = r'/cephfs2/srogers/DEEPLABCUT_models/LMDC_model_videos/models/LMDC-StefanRC-2025-03-11/config.yaml'

# Global flag for the malloc_trim thread
stop_trim_thread = False


def malloc_trim_thread(interval=15):
    """
    Periodically call malloc_trim to return freed memory to the OS.
    This prevents memory fragmentation issues during long video analysis.
    """
    global stop_trim_thread
    
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        print("Warning: malloc_trim not available (Linux only). Memory usage may be higher.")
        return
    
    while not stop_trim_thread:
        time.sleep(interval)
        if not stop_trim_thread:
            # Trim memory and run garbage collection
            libc.malloc_trim(0)
            gc.collect()


def analyse(video_path, gpu_id):
    """
    Analyse video with malloc_trim fix for memory issues.
    
    Args:
        video_path: Path to video file
        gpu_id: GPU device ID to use
    """
    global stop_trim_thread
    stop_trim_thread = False
    
    # Start malloc_trim thread
    trim_thread = threading.Thread(target=malloc_trim_thread, args=(15,), daemon=True)
    trim_thread.start()
    
    # Run analysis
    start_time = time.perf_counter()
    print(f"Analysing {str(video_path)}")
    print(f"Using GPU {gpu_id}")
    
    try:
        analyze_videos(
            config=config,
            videos=[video_path], 
            videotype='.avi', 
            save_as_csv=True, 
            gputouse=gpu_id
        )
        
        # Calculate duration
        elapsed_time = time.perf_counter() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        
        print(f"Analysis of {str(video_path)} complete.")
        print(f"Duration: {minutes} minutes and {seconds:.2f} seconds")
        
    finally:
        # Stop malloc_trim thread
        stop_trim_thread = True


if __name__ == "__main__":
    # Optional: Use command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if len(sys.argv) > 2:
            gpu_id = int(sys.argv[2])
    
    analyse(video_path, gpu_id)