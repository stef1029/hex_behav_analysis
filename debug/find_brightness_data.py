import os
import sys
import json
from pathlib import Path
from datetime import datetime

from debug_cohort_folder import Cohort_folder
from analyse_video import analyze_video_with_roi_parallel

def find_video_file(session_folder):
    """
    Look for the main video file in the session folder.
    Returns the path if found, None otherwise.
    """
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv']
    
    # Try to find a video file in the session folder
    session_path = Path(session_folder)
    for ext in video_extensions:
        # Look for files matching session_id pattern with this extension
        potential_videos = list(session_path.glob(f"*{ext}"))
        if potential_videos:
            # Return the first match
            return str(potential_videos[0])
    
    # If we get here, no video file was found
    return None

def validate_roi_file(roi_path):
    """
    Checks if the ROI file contains the required x,y coordinates.
    Returns True if valid, False otherwise.
    """
    try:
        with open(roi_path, "r") as f:
            roi_data = json.load(f)
            
        # Check for required fields - only x and y are needed
        required_fields = ["x", "y"]
        
        if all(field in roi_data for field in required_fields):
            # Don't add width and height - we're analyzing single pixels now
            return True
        else:
            print(f"ROI file {roi_path} missing required fields {required_fields}")
            return False
            
    except Exception as e:
        print(f"Error validating ROI file {roi_path}: {e}")
        return False

def process_session(session_folder, session_id, num_processes=None):
    """
    Process a single session - find video, validate ROI, run analysis.
    Returns (session_id, success, error_message)
    """
    print(f"Processing session {session_id}...")
    
    try:
        # Find the truncated_start_report folder
        truncated_folder = Path(session_folder) / "truncated_start_report"
        roi_path = truncated_folder / f"{session_id}_roi.json"
        
        # Validate ROI file
        if not roi_path.is_file():
            return (session_id, False, f"ROI file not found: {roi_path}")
        
        if not validate_roi_file(roi_path):
            return (session_id, False, "Invalid ROI file format")
        
        # Find video file
        video_path = find_video_file(session_folder)
        if not video_path:
            return (session_id, False, "No video file found in session folder")
        
        # Run the analysis
        num_frames = analyze_video_with_roi_parallel(
            session_folder=str(session_folder),
            session_id=session_id,
            video_path=video_path,
            num_processes=num_processes
        )
        
        # Check if output file was created
        output_path = truncated_folder / f"{session_id}_brightness_data.csv"
        if output_path.is_file():
            return (session_id, True, f"Analysis complete, processed {num_frames} frames")
        else:
            return (session_id, False, "Analysis ran but no output file was created")
            
    except Exception as e:
        return (session_id, False, f"Error: {str(e)}")

def batch_analyze_cohort(cohort_dir, num_processes=None, specific_session=None):
    """
    Main function to analyze all sessions in a cohort with ROI data.
    Processes one video at a time, but uses parallelism within each video.
    For each video, extracts brightness data from the single pixel at ROI coordinates.
    
    Args:
        cohort_dir: Path to the cohort directory
        num_processes: Number of processes to use for each video (None = auto)
        specific_session: If provided, only analyze this specific session ID
    """
    # Create a cohort object
    cohort = Cohort_folder(
        cohort_directory=cohort_dir,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False
    )
    
    # Track sessions and results
    sessions_to_process = []
    
    # Find all sessions with ROI data
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            # Skip if we're looking for a specific session and this isn't it
            if specific_session and session_id != specific_session:
                continue
                
            if sdict.get("has_truncated_start_report", False):
                session_folder = Path(sdict["directory"])
                tsr_folder = session_folder / "truncated_start_report"
                roi_path = tsr_folder / f"{session_id}_roi.json"
                
                if roi_path.is_file():
                    # Found a session with ROI data
                    sessions_to_process.append((session_folder, session_id))
    
    print(f"Found {len(sessions_to_process)} sessions with ROI data to process")
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(cohort_dir) / f"analysis_log_{timestamp}.txt"
    
    # Process sessions one at a time
    results = []
    for i, (session_folder, session_id) in enumerate(sessions_to_process):
        print(f"\nProcessing session {i+1}/{len(sessions_to_process)}: {session_id}")
        
        try:
            # Process each session one at a time (but with internal parallelism)
            result = process_session(session_folder, session_id, num_processes)
            results.append(result)
            
            # Log result
            success_str = "SUCCESS" if result[1] else "FAILED"
            log_message = f"{session_id}: {success_str} - {result[2]}"
            print(log_message)
            
            with open(log_path, "a") as log_file:
                log_file.write(f"{log_message}\n")
                
        except Exception as e:
            print(f"{session_id}: ERROR - {str(e)}")
            with open(log_path, "a") as log_file:
                log_file.write(f"{session_id}: ERROR - {str(e)}\n")
    
    # Summarize results
    successful = sum(1 for r in results if r[1])
    print(f"\nAnalysis complete: {successful}/{len(results)} sessions processed successfully")
    print(f"Log file saved to: {log_path}")
    
    return results

def main():
    """
    Main function with hardcoded parameters.
    Modify these values directly instead of using command line arguments.
    """
    # Set your parameters here
    cohort_dir = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"  # Path to your cohort directory
    num_processes = 8                                            # Number of processes for each video
    specific_session = None                                      # Set to a session ID to process only that session
    
    # Run the analysis
    batch_analyze_cohort(
        cohort_dir=cohort_dir,
        num_processes=num_processes,
        specific_session=specific_session
    )

if __name__ == "__main__":
    main()