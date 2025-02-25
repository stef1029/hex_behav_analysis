import os
import cv2
import math
import json
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

def _process_chunk(args):
    """
    Worker function for multiprocessing.
    Processes a specific chunk of frames to detect threshold crossings.
    """
    (video_path,
     start_frame,
     end_frame,
     x, y, w, h,
     brightness_threshold,
     fps) = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Seek to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    was_above_threshold = False
    timestamps = []

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop ROI and convert to grayscale
        roi_frame = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        avg_brightness = np.mean(gray_roi)

        # Check threshold crossing (below -> above)
        if avg_brightness >= brightness_threshold and not was_above_threshold:
            timestamp_ms = frame_idx * (1000.0 / fps)
            timestamps.append(timestamp_ms)
            was_above_threshold = True
        elif avg_brightness < brightness_threshold and was_above_threshold:
            was_above_threshold = False

    cap.release()
    return timestamps


def analyze_video_with_roi_parallel(session_folder,
                                    session_id,
                                    video_path,
                                    brightness_threshold=150.0,
                                    num_processes=None):
    """
    Loads video, retrieves ROI data from JSON, and runs parallel threshold analysis.
    Saves the resulting timestamps to <session_id>_touch_timestamps.txt in the
    'truncated_start_report' folder.
    
    Args:
        session_folder (str): Path to the session folder
        session_id (str): Session identifier
        video_path (str): Path to the video file
        brightness_threshold (float): Brightness threshold to detect crossings
        num_processes (int, optional): Number of processes to use. Default: all CPU cores
    
    Returns:
        None
    """
    if num_processes is None:
        num_processes = cpu_count()

    truncated_folder = os.path.join(session_folder, "truncated_start_report")
    roi_file_path = os.path.join(truncated_folder, f"{session_id}_roi.json")

    # Load ROI from JSON
    if not os.path.isfile(roi_file_path):
        raise FileNotFoundError(f"ROI file not found: {roi_file_path}")

    with open(roi_file_path, "r") as f:
        roi_data = json.load(f)

    # Check if we need to provide default width and height
    if "x" in roi_data and "y" in roi_data:
        x = roi_data["x"]
        y = roi_data["y"]
        
        # Default width and height if not present
        if "w" not in roi_data:
            w = 50  # Default width
        else:
            w = roi_data["w"]
            
        if "h" not in roi_data:
            h = 50  # Default height
        else:
            h = roi_data["h"]
    else:
        raise ValueError(f"ROI data missing required x,y coordinates: {roi_file_path}")

    # Open video to fetch total frames & fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video with {total_frames} frames at {fps} fps")
    print(f"Using ROI: x={x}, y={y}, w={w}, h={h}")
    print(f"Brightness threshold: {brightness_threshold}")
    
    cap.release()

    if total_frames <= 0:
        raise ValueError("Video has no frames or metadata is missing.")
    if fps <= 0:
        raise ValueError("Invalid FPS in video metadata; can't compute timestamps.")

    # Prepare chunked processing
    chunk_size = math.ceil(total_frames / num_processes)
    jobs = []
    for i in range(num_processes):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)
        if start_frame >= total_frames:
            break

        jobs.append((
            video_path,
            start_frame,
            end_frame,
            x,
            y,
            w,
            h,
            brightness_threshold,
            fps
        ))

    print(f"Splitting video into {len(jobs)} chunks for parallel processing")

    # Parallel processing
    crossing_timestamps = []
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_chunk, jobs)
        for chunk_result in results:
            crossing_timestamps.extend(chunk_result)

    # Sort timestamps in ascending order
    crossing_timestamps.sort()

    # Save timestamps to a file
    output_path = os.path.join(truncated_folder, f"{session_id}_touch_timestamps.txt")
    with open(output_path, "w") as f:
        for t in crossing_timestamps:
            f.write(f"{t:.2f}\n")

    print(f"Analysis complete. Found {len(crossing_timestamps)} threshold crossings.")
    print(f"Timestamps saved in: {output_path}")
    
    return len(crossing_timestamps)