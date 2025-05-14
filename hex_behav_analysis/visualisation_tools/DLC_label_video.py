import deeplabcut
import multiprocessing
import os
from pathlib import Path
from tqdm import tqdm

def process_single_video(args):
    """
    Worker function to process a single video with DeepLabCut.
    
    Args:
        args: Tuple containing (video_path, config_path)
        
    Returns:
        Tuple of (video_path, success_status)
    """
    video_path, config_path = args
    try:
        deeplabcut.create_labeled_video(
            config_path,
            [str(video_path)],
            videotype=os.path.splitext(str(video_path))[1],
            draw_skeleton=True,
            trailpoints=0,
            displaycropped=False,
            overwrite=True
        )
        return video_path, True
    except Exception as e:
        return video_path, False, str(e)

def process_sessions(sessions, config_path, num_workers=None):
    """
    Process multiple session videos in parallel using DeepLabCut.
    
    Args:
        sessions: List of Session objects with session_video attributes
        config_path: Path to the DeepLabCut config file
        num_workers: Number of parallel processes to use (default: None, which uses CPU count)
        
    Returns:
        List of (video_path, success_status) tuples
    """
    if num_workers is None:
        # Use number of CPU cores minus 1 to avoid overloading the system
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Extract video paths from session objects
    video_paths = []
    for session in sessions:
        try:
            # Ensure session_video is a Path object or convert to one
            if isinstance(session.session_video, Path):
                video_path = session.session_video
            else:
                video_path = Path(session.session_video)
                
            if video_path.exists():
                video_paths.append(video_path)
            else:
                print(f"Warning: Video file not found: {video_path}")
        except AttributeError as e:
            print(f"Error accessing session_video for session {session.session_ID if hasattr(session, 'session_ID') else 'unknown'}: {e}")
    
    if not video_paths:
        print("No valid video paths found in the provided sessions.")
        return []
    
    print(f"Processing {len(video_paths)} videos using {num_workers} parallel workers")
    
    # Create argument tuples for each video
    args_list = [(video_path, config_path) for video_path in video_paths]
    
    # Process videos in parallel
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress
        for result in tqdm(pool.imap_unordered(process_single_video, args_list), 
                           total=len(args_list),
                           desc="Processing videos"):
            results.append(result)
    
    # Summarize results
    successful = sum(1 for _, status, *_ in results if status)
    print(f"Processing complete: {successful}/{len(results)} videos successfully processed")
    
    # Print errors for failed videos
    for video_path, status, *error_info in results:
        if not status:
            print(f"Failed to process {video_path}: {error_info[0] if error_info else 'Unknown error'}")
    
    return results

def process_video_list(video_paths, config_path, num_workers=None):
    """
    Process a list of video paths in parallel using DeepLabCut.
    
    Args:
        video_paths: List of paths to video files
        config_path: Path to the DeepLabCut config file
        num_workers: Number of parallel processes to use (default: None, which uses CPU count)
        
    Returns:
        List of (video_path, success_status) tuples
    """
    if num_workers is None:
        # Use number of CPU cores minus 1 to avoid overloading the system
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Validate video paths
    valid_paths = []
    for path in video_paths:
        video_path = Path(path)
        if video_path.exists():
            valid_paths.append(video_path)
        else:
            print(f"Warning: Video file not found: {path}")
    
    if not valid_paths:
        print("No valid video paths found.")
        return []
    
    print(f"Processing {len(valid_paths)} videos using {num_workers} parallel workers")
    
    # Create argument tuples for each video
    args_list = [(video_path, config_path) for video_path in valid_paths]
    
    # Process videos in parallel
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress
        for result in tqdm(pool.imap_unordered(process_single_video, args_list), 
                           total=len(args_list),
                           desc="Processing videos"):
            results.append(result)
    
    # Summarize results
    successful = sum(1 for _, status, *_ in results if status)
    print(f"Processing complete: {successful}/{len(results)} videos successfully processed")
    
    # Print errors for failed videos
    for video_path, status, *error_info in results:
        if not status:
            print(f"Failed to process {video_path}: {error_info[0] if error_info else 'Unknown error'}")
    
    return results


# Example usage when imported:
# from dlc_processor import process_sessions
# 
# # With session objects:
# process_sessions(my_sessions, config_path="/path/to/config.yaml")
#
# # With video paths:
# process_video_list(["/path/to/video1.avi", "/path/to/video2.mp4"], config_path="/path/to/config.yaml")


# Example usage as script:
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Process videos with DeepLabCut")
    parser.add_argument("--config", required=True, help="Path to DeepLabCut config file")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    process_video_list(args.videos, args.config, args.workers)