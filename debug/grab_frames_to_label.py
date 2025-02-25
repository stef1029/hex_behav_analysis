import sys
import os
import cv2
import json
from pathlib import Path
from debug_cohort_folder import Cohort_folder

def extract_middle_frame(session_folder, session_id, video_path):
    """
    Opens 'video_path', seeks to the middle frame, and saves it as
    <session_id>_middle_frame.png in [session_folder]/truncated_start_report/.
    """
    session_path = Path(session_folder)
    report_folder = session_path / "truncated_start_report"
    report_folder.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise IOError(f"Video has no frames or metadata is missing.")

    middle_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Failed to read the middle frame from the video.")

    out_path = report_folder / f"{session_id}_middle_frame.png"
    cv2.imwrite(str(out_path), frame)
    print(f"Saved middle frame to: {out_path}")

def get_frame_count_from_json(session_folder):
    """
    Reads truncated_start_info.json and extracts the total number of frames
    from the crash_details field.
    
    Returns:
        int: Number of frames, or 0 if information not found
    """
    json_path = Path(session_folder) / "truncated_start_report" / "truncated_start_info.json"
    
    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return 0
    
    try:
        with open(json_path, 'r') as f:
            info = json.load(f)
        
        # Extract frame count from crash_details field
        if "crash_details" in info:
            crash_details = info["crash_details"]
            if "Total frames:" in crash_details:
                # Extract the number from "Total frames: X (multiple of 200: Y)"
                frames_str = crash_details.split("Total frames:")[1].split("(")[0].strip()
                return int(frames_str)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
    
    return 0

def main():
    """
    Usage:
        python extract_middle_frames_cohort.py <cohort_directory>

    This script:
    1) Initializes the Cohort_folder with the given directory.
    2) Iterates over all sessions in the cohort.
    3) For each session that has a truncated_start_report folder:
       a) Checks truncated_start_info.json for frame count
       b) If frame count >= 18000, retrieves the raw video path and extracts the middle frame
    4) Saves the extracted frame to [session_folder]/truncated_start_report/<session_id>_middle_frame.png.
    """

    cohort_directory = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"

    # Instantiate the cohort
    cohort = Cohort_folder(
        cohort_directory,
        multi=True,               # or False, depending on your structure
        portable_data=False,      # or True, if you're using portable data
        OEAB_legacy=True,         # or False, depends on your data layout
        ignore_tests=True,        # depends on your preference
        use_existing_cohort_info=False
    )

    # Now iterate over all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if sdict.get("has_truncated_start_report", False):
                # Check frame count in the JSON file
                frame_count = get_frame_count_from_json(sdict["directory"])
                
                if frame_count < 18000:
                    print(f"Skipping session {session_id} - only has {frame_count} frames (< 18000)")
                    continue
                
                # If the frame count is sufficient, process the video
                video_path = None

                # Example: non-portable raw_data approach
                if "raw_data" in sdict and sdict["raw_data"].get("raw_video") != "None":
                    video_path = sdict["raw_data"]["raw_video"]

                if video_path and Path(video_path).is_file():
                    try:
                        extract_middle_frame(
                            session_folder=sdict["directory"],
                            session_id=session_id,
                            video_path=video_path
                        )
                        print(f"Processed session {session_id} with {frame_count} frames")
                    except Exception as e:
                        print(f"Error extracting middle frame for session {session_id}:\n{e}")
                else:
                    print(f"No valid raw video found for session {session_id}, skipping.")

if __name__ == "__main__":
    main()