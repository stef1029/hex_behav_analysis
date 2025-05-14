from pathlib import Path
import cv2 as cv
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def draw_LEDs(session, output_path, start=0, end=None, condense_to_trials=False, buffer_frames=30, trial_range=None):
    """
    Create a video with LED markers shown only during cue activation time,
    and mouse position and bearing visualized for the entire trial duration.
    
    Args:
        session: An initialised Session object containing trial data.
        output_path: Path where the output video will be saved.
        start: Starting frame index for processing (default: 0).
        end: Ending frame index for processing (default: None, which means process all frames).
        condense_to_trials: If True, only include frames within trials (plus buffer) in the output video.
        buffer_frames: Number of frames to include before and after each trial when condensing.
        trial_range: Optional tuple of (start_trial_index, end_trial_index) to specify which trials to include.
                     If None, all trials are included. Trial indices are 0-based.
    
    Returns:
        Path to the created video file.
    """
    # Determine the port angles based on rig ID
    if session.rig_id == 1:
        port_angles = [64, 124, 184, 244, 304, 364]  # calibrated 14/2/24
    elif session.rig_id == 2:
        port_angles = [240, 300, 360, 420, 480, 540]
    else:
        raise ValueError(f"Invalid rig ID: {session.rig_id}")

    # Prepare frame annotations dictionary
    # Structure: {frame_index: {"cue": port_num, "trial_num": num, "show_cue": bool, "midpoint": (x,y), "bearing": angle}}
    frame_annotations = {}
    
    # Track trial frame ranges for video condensing
    trial_frame_ranges = []
    
    # Process each trial to extract relevant info
    for i, trial in enumerate(session.trials):
        # Skip trials that are outside the specified trial range
        if trial_range is not None:
            start_trial, end_trial = trial_range
            if i < start_trial or i > end_trial:
                continue
                
        # Skip trials without video frames or DLC data
        if not trial.get("video_frames") or trial.get("DLC_data") is None:
            continue
            
        # Get timestamps for trial duration and cue periods
        if 'timestamps' not in trial["DLC_data"].columns:
            continue
            
        # Get trial timestamps and video frames
        video_frames = trial["video_frames"]
        
        # Save frame range for this trial (for video condensing)
        if len(video_frames) > 0:
            trial_frame_ranges.append((video_frames[0], video_frames[-1]))
        
        timestamps = trial["DLC_data"]['timestamps'].values
        
        # Get cue timing information
        cue_start_time = trial.get("cue_start")
        cue_end_time = trial.get("cue_end")
        
        # If cue_end doesn't exist, try to infer it
        if cue_end_time is None:
            # Check if there's a next_sensor timestamp (when the mouse interacted)
            if "next_sensor" in trial and isinstance(trial["next_sensor"], dict) and "sensor_start" in trial["next_sensor"]:
                cue_end_time = trial["next_sensor"]["sensor_start"]
            # Otherwise use the start of the next trial or the end of the session
            elif i + 1 < len(session.trials):
                cue_end_time = session.trials[i + 1].get("cue_start")
            else:
                cue_end_time = session.last_timestamp
                
        # Get trial start and end times
        trial_start_time = trial.get("cue_start")  # Using cue_start as trial start
        if i + 1 < len(session.trials):
            trial_end_time = session.trials[i + 1].get("cue_start")
        else:
            trial_end_time = session.last_timestamp
        
        # Get the mouse midpoint and bearing data
        turn_data = trial.get("turn_data")
            
        # Process each frame in this trial
        for idx, frame_idx in enumerate(video_frames):
            if idx < len(timestamps):
                timestamp = timestamps[idx]
                
                # Skip frames without proper indices
                if idx >= len(timestamps):
                    continue
                    
                # Determine if we should show the cue for this frame
                show_cue = cue_start_time <= timestamp <= cue_end_time
                
                # Get mouse position and bearing for this frame if available
                midpoint = None
                bearing = None
                if turn_data is not None:
                    midpoint = turn_data.get("midpoint")
                    bearing = turn_data.get("bearing")
                
                # Add the annotation for this frame
                frame_annotations[frame_idx] = {
                    "text": f"trial {i + 1}, cue: {trial['correct_port']}",
                    "cue": trial['correct_port'],
                    "trial_num": i + 1,
                    "show_cue": show_cue,
                    "midpoint": midpoint,
                    "bearing": bearing
                }

    # Set up output paths
    output_folder = Path(output_path) / "drawn_videos"
    
    # Create appropriate filename suffix based on parameters
    filename_suffix = ""
    if condense_to_trials:
        filename_suffix = "_condensed"
        if trial_range is not None:
            start_trial, end_trial = trial_range
            filename_suffix += f"_trials_{start_trial+1}-{end_trial+1}"  # +1 for 1-based trial numbering in output
            
    filename = f"{session.session_ID}_labelled_LEDs_with_mouse{filename_suffix}.mp4"
    output_filename = output_folder / filename

    # Ensure the output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(session.session_video))
    if not cap.isOpened():
        print(f"Error: Could not open video file {session.session_video}")
        return None

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Width: {frame_width}, Height: {frame_height}, frame count: {total_frame_count}")
    
    # Determine which frames to process based on condense_to_trials parameter
    frame_indices_to_process = []
    
    if condense_to_trials and trial_frame_ranges:
        # Add frames from each trial plus buffer
        for start_frame, end_frame in trial_frame_ranges:
            buffer_start = max(0, start_frame - buffer_frames)
            buffer_end = min(total_frame_count - 1, end_frame + buffer_frames)
            frame_indices_to_process.extend(range(buffer_start, buffer_end + 1))
        
        # Remove duplicates and sort
        frame_indices_to_process = sorted(list(set(frame_indices_to_process)))
        
        # When condensing, we ignore the start and end parameters completely
        # as the trial_range parameter controls what's included
    else:
        # Process all frames between start and end
        if end is None:
            end = total_frame_count
        end = min(end, total_frame_count)
        frame_indices_to_process = list(range(start, end))
    
    # Create the output video writer
    out = cv.VideoWriter(str(output_filename), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: VideoWriter failed to open.")
        return None
    
    centre = (int(frame_width / 2), int(frame_height / 2))
    cue_position = (frame_height / 2) - 25
    
    # Arrow parameters for bearing visualization
    arrow_length = 50  # Length of the arrow in pixels
    arrow_color = (0, 0, 255)  # Red color in BGR
    arrow_thickness = 2
    
    # Use tqdm for progress tracking
    for frame_index in tqdm(frame_indices_to_process, desc="Processing frames"):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame at index {frame_index}")
            continue  # Skip this frame instead of breaking the loop

        # Add trial timestamp to all frames
        cv.putText(
            frame, f"Frame: {frame_index}", (50, frame_height - 20), 
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA
        )

        # Check if current frame has annotations
        if frame_index in frame_annotations:
            annotation = frame_annotations[frame_index]
            
            # Add trial and cue text
            frame = cv.putText(
                frame, annotation["text"], (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv.LINE_AA
            )
            
            # Draw cue LED if it should be visible at this frame
            if annotation["show_cue"]:
                port = annotation["cue"]
                try:
                    port = int(port) - 1
                    port_angle = port_angles[port]
                    x2 = int(centre[0] + cue_position * math.cos(math.radians(port_angle)))
                    y2 = int(centre[1] - cue_position * math.sin(math.radians(port_angle)))
                    # Draw a marker at the cue position
                    frame = cv.drawMarker(
                        frame, (x2, y2), (0, 255, 0), markerType=cv.MARKER_STAR,
                        markerSize=30, thickness=2, line_type=cv.LINE_AA
                    )
                except (ValueError, TypeError, IndexError):
                    if port == "audio-1" or port == "audio":
                        frame = cv.putText(frame, "audio", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            
            # Draw mouse position and bearing if available
            if annotation["midpoint"] is not None and annotation["bearing"] is not None:
                midpoint = annotation["midpoint"]
                bearing = annotation["bearing"]
                
                # Convert midpoint to integers
                midpoint = (int(midpoint[0]), int(midpoint[1]))
                
                # Draw a circle at the mouse midpoint
                cv.circle(frame, midpoint, 5, (255, 0, 0), -1)  # Blue circle
                
                # Calculate endpoint for the bearing arrow
                bearing_rad = math.radians(bearing)
                end_x = int(midpoint[0] + arrow_length * math.cos(bearing_rad))
                end_y = int(midpoint[1] - arrow_length * math.sin(bearing_rad))  # Subtract because y increases downward
                
                # Draw the arrow
                cv.arrowedLine(frame, midpoint, (end_x, end_y), arrow_color, arrow_thickness, tipLength=0.3)

        # Write the processed frame to the output
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    
    # Report statistics
    if condense_to_trials:
        original_duration = total_frame_count / fps
        condensed_duration = len(frame_indices_to_process) / fps
        print(f"Original video duration: {original_duration:.2f}s")
        print(f"Condensed video duration: {condensed_duration:.2f}s")
        print(f"Compression ratio: {original_duration/condensed_duration:.2f}x")
        
        if trial_range is not None:
            start_trial, end_trial = trial_range
            print(f"Included trials: {start_trial+1} to {end_trial+1}")
    
    print(f"Processing complete. Video saved to {output_filename}")
    return output_filename