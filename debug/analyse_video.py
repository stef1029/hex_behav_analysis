#!/usr/bin/env python3

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
    Processes a specific chunk of frames and outputs brightness data for each frame.
    Now focuses on a single pixel rather than a rectangular ROI.
    """
    (video_path,
     start_frame,
     end_frame,
     x, y,
     fps) = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Seek to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # List to store frame data: (frame_number, timestamp_ms, brightness)
    brightness_data = []

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract single pixel value at (x,y)
        try:
            # Ensure point is within frame bounds
            height, width = frame.shape[:2]
            if 0 <= x < width and 0 <= y < height:
                pixel = frame[y, x]
                
                # Convert BGR pixel to grayscale (0.299*R + 0.587*G + 0.114*B)
                brightness = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
            else:
                print(f"Warning: Point ({x},{y}) is outside frame bounds ({width}x{height}). Using border pixel.")
                # Use the closest pixel at the border
                x_bounded = max(0, min(x, width-1))
                y_bounded = max(0, min(y, height-1))
                pixel = frame[y_bounded, x_bounded]
                brightness = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
        except Exception as e:
            print(f"Error processing frame {frame_idx} at point ({x},{y}): {e}")
            brightness = 0
        
        # Calculate timestamp in milliseconds
        timestamp_ms = frame_idx * (1000.0 / fps)
        
        # Store frame number, timestamp, and brightness
        brightness_data.append((frame_idx, timestamp_ms, brightness))

    cap.release()
    return brightness_data


def analyze_video_with_roi_parallel(session_folder,
                                    session_id,
                                    video_path,
                                    num_processes=None):
    """
    Loads video, retrieves ROI data from JSON, and runs parallel analysis to output
    brightness data for each frame. Now analyzes only a single pixel's brightness.
    Saves the brightness data to a CSV file in the 'truncated_start_report' folder.
    
    Args:
        session_folder (str): Path to the session folder
        session_id (str): Session identifier
        video_path (str): Path to the video file
        num_processes (int, optional): Number of processes to use. Default: all CPU cores
    
    Returns:
        int: Number of frames processed
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

    # Check for required coordinates
    if "x" in roi_data and "y" in roi_data:
        x = roi_data["x"]
        y = roi_data["y"]
    else:
        raise ValueError(f"ROI data missing required x,y coordinates: {roi_file_path}")

    # Open video to fetch total frames & fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video with {total_frames} frames at {fps} fps")
    print(f"Using pixel at point: x={x}, y={y}")
    
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
            fps
        ))

    print(f"Splitting video into {len(jobs)} chunks for parallel processing")

    # Parallel processing
    all_brightness_data = []
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_chunk, jobs)
        for chunk_result in results:
            all_brightness_data.extend(chunk_result)

    # Sort data by frame number
    all_brightness_data.sort(key=lambda x: x[0])

    # Save brightness data to a CSV file
    output_path = os.path.join(truncated_folder, f"{session_id}_brightness_data.csv")
    with open(output_path, "w") as f:
        # Write header
        f.write("frame,timestamp_ms,brightness\n")
        # Write data rows
        for frame_idx, timestamp, brightness in all_brightness_data:
            f.write(f"{frame_idx},{timestamp:.2f},{brightness:.2f}\n")

    print(f"Analysis complete. Processed {len(all_brightness_data)} frames.")
    print(f"Brightness data saved in: {output_path}")
    
    return len(all_brightness_data)


