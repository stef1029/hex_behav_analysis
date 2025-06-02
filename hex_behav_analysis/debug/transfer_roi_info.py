#!/usr/bin/env python3

import sys
import shutil
import os
import json
from pathlib import Path

from debug_cohort_folder import Cohort_folder
# ^ Adjust this import if your Cohort_folder class is defined elsewhere.

def transfer_roi_back(source_dir, original_cohort_dir):
    """
    Transfers ROI JSON files from the source directory back to their original locations
    in the cohort directory.
    
    1) Builds a Cohort_folder to read all session metadata from the original cohort.
    2) For each session that has a 'truncated_start_report' folder, looks for 
       corresponding ROI JSON files in the source directory and copies them back.
    
    Args:
        source_dir (str): Directory containing the labeled ROI JSON files
        original_cohort_dir (str): Original cohort directory where files should be copied back to
    """
    # 1) Instantiate the cohort for the original data
    cohort = Cohort_folder(
        cohort_directory=original_cohort_dir,
        multi=True,                # or False, depending on your data structure
        portable_data=False,       # or True, if you're using portable data
        OEAB_legacy=True,          # depends on your folder structure
        ignore_tests=True,         # or False, depending on your use case
        use_existing_cohort_info=False
    )

    original_cohort_path = Path(original_cohort_dir).resolve()
    source_path = Path(source_dir).resolve()

    # Track statistics
    total_sessions = 0
    sessions_with_tsr = 0
    roi_files_found = 0
    roi_files_copied = 0
    roi_files_skipped = 0

    # 2) Iterate over all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            total_sessions += 1
            
            if sdict.get("has_truncated_start_report", False):
                sessions_with_tsr += 1
                
                # Get the original session folder and its truncated_start_report
                original_session_folder = Path(sdict["directory"])
                original_tsr_folder = original_session_folder / "truncated_start_report"
                
                # Build the path to where this session should be in the source directory
                try:
                    session_subpath = original_session_folder.relative_to(original_cohort_path)
                except ValueError:
                    print(f"Session folder {original_session_folder} is not relative to {original_cohort_path}, skipping.")
                    continue
                
                # Source folder where labeled data should be
                source_session_folder = source_path / session_subpath
                source_tsr_folder = source_session_folder / "truncated_start_report"
                
                # Check for ROI JSON file
                roi_json_path = source_tsr_folder / f"{session_id}_roi.json"
                
                if roi_json_path.is_file():
                    roi_files_found += 1
                    
                    # Destination path in the original cohort
                    dest_roi_path = original_tsr_folder / f"{session_id}_roi.json"
                    
                    # Check if file already exists at destination (optional overwrite logic)
                    if dest_roi_path.is_file():
                        print(f"ROI file already exists at {dest_roi_path}. Overwriting.")
                    
                    try:
                        # Make sure the destination directory exists
                        original_tsr_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Copy the ROI JSON file
                        shutil.copy2(roi_json_path, dest_roi_path)
                        print(f"Copied ROI from {roi_json_path} to {dest_roi_path}")
                        roi_files_copied += 1
                        
                        # Verify JSON data is valid by loading it (optional validation)
                        try:
                            with open(dest_roi_path, 'r') as f:
                                roi_data = json.load(f)
                                if "x" not in roi_data or "y" not in roi_data:
                                    print(f"Warning: ROI data seems incomplete in {dest_roi_path}")
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON in {dest_roi_path}")
                            
                    except Exception as e:
                        print(f"Error copying {roi_json_path} -> {dest_roi_path}: {str(e)}")
                        roi_files_skipped += 1
                else:
                    print(f"No ROI file found at {roi_json_path}, skipping.")

    # Print summary
    print(f"\nSummary:")
    print(f"Total sessions: {total_sessions}")
    print(f"Sessions with truncated_start_report: {sessions_with_tsr}")
    print(f"ROI files found: {roi_files_found}")
    print(f"ROI files successfully copied: {roi_files_copied}")
    print(f"ROI files skipped due to errors: {roi_files_skipped}")

def main():
    """
    Usage:
        python transfer_roi_back.py <source_directory> <original_cohort_directory>
    Example:
        python transfer_roi_back.py /data/TruncatedReportsOnly /data/OriginalCohort
    """
    # Replace these with your actual directories
    source_dir = r"/cephfs2/srogers/debug_vids/September_cohort_label_frames"
    original_cohort_dir = r"/cephfs2/srogers/Behaviour code/2409_September_cohort/DATA_ArduinoDAQ"
    
    # You can uncomment the following to use command-line arguments instead
    # if len(sys.argv) != 3:
    #     print(f"Usage: {sys.argv[0]} <source_directory> <original_cohort_directory>")
    #     sys.exit(1)
    # source_dir = sys.argv[1]
    # original_cohort_dir = sys.argv[2]

    transfer_roi_back(source_dir, original_cohort_dir)

if __name__ == "__main__":
    main()