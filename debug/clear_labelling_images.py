import os
from pathlib import Path
from debug_cohort_folder import Cohort_folder

def clear_middle_frame_images(cohort_directory):
    """
    Clears all middle frame images from truncated_start_report folders in the cohort.
    
    Args:
        cohort_directory: Path to the cohort directory
    """
    print(f"Clearing middle frame images from cohort: {cohort_directory}")
    
    # Instantiate the cohort
    cohort = Cohort_folder(
        cohort_directory,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False
    )
    
    total_removed = 0
    sessions_processed = 0
    
    # Iterate through all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if sdict.get("has_truncated_start_report", False):
                session_folder = Path(sdict["directory"])
                report_folder = session_folder / "truncated_start_report"
                
                if report_folder.exists():
                    # Look for image files matching the pattern
                    image_files = list(report_folder.glob("*_middle_frame.png"))
                    
                    for img_file in image_files:
                        try:
                            img_file.unlink()  # Delete the file
                            print(f"Removed: {img_file}")
                            total_removed += 1
                        except Exception as e:
                            print(f"Error removing {img_file}: {e}")
                    
                    sessions_processed += 1
    
    print(f"Completed: Removed {total_removed} image files from {sessions_processed} sessions.")

def main():
    """
    Script to clear all middle frame images from the truncated_start_report folders.
    
    Usage:
        python clear_middle_frames.py
    """
    cohort_directory = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"
    
    # Ask for confirmation before proceeding
    print(f"WARNING: This will delete all middle frame images from {cohort_directory}")
    confirmation = input("Are you sure you want to proceed? (y/n): ")
    
    if confirmation.lower() == 'y':
        clear_middle_frame_images(cohort_directory)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()