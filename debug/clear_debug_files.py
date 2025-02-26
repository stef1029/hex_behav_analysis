import os
from pathlib import Path
from debug_cohort_folder import Cohort_folder

def clear_files(cohort_directory, file_type='images', dry_run=False):
    """
    Clears specified file types from truncated_start_report folders in the cohort.
    
    Args:
        cohort_directory: Path to the cohort directory
        file_type: Type of files to clear - 'images' or 'roi' or 'brightness'
        dry_run: If True, only print what would be deleted without actually deleting
    """
    if file_type == 'images':
        print(f"Clearing middle frame images from cohort: {cohort_directory}")
        file_pattern = "*_middle_frame.png"
        file_description = "image files"
    elif file_type == 'roi':
        print(f"Clearing ROI JSON files from cohort: {cohort_directory}")
        file_pattern = "*_roi.json"
        file_description = "ROI files"
    elif file_type == 'brightness':
        print(f"Clearing brightness data files from cohort: {cohort_directory}")
        file_pattern = "*_brightness_data.csv"
        file_description = "brightness data files"
    else:
        raise ValueError(f"Unknown file_type: {file_type}. Use 'images', 'roi', or 'brightness'.")
    
    # Add dry run notification
    if dry_run:
        print("[DRY RUN] No files will actually be deleted")
    
    # Instantiate the cohort
    cohort = Cohort_folder(
        cohort_directory,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False
    )
    
    total_found = 0
    total_removed = 0
    sessions_processed = 0
    
    # Iterate through all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if sdict.get("has_truncated_start_report", False):
                session_folder = Path(sdict["directory"])
                report_folder = session_folder / "truncated_start_report"
                
                if report_folder.exists():
                    # Look for files matching the pattern
                    target_files = list(report_folder.glob(file_pattern))
                    total_found += len(target_files)
                    
                    for file_path in target_files:
                        try:
                            if not dry_run:
                                file_path.unlink()  # Delete the file
                                total_removed += 1
                            print(f"{'Would remove' if dry_run else 'Removed'}: {file_path}")
                        except Exception as e:
                            print(f"Error handling {file_path}: {e}")
                    
                    sessions_processed += 1
    
    action_word = "Found" if dry_run else "Removed"
    print(f"Completed: {action_word} {total_found} {file_description} from {sessions_processed} sessions.")
    
    # If dry run, show how many would have been deleted
    if dry_run:
        print(f"In a real run, {total_found} files would be deleted.")

def main():
    """
    Script to clear files from the truncated_start_report folders.
    
    Usage:
        python clear_files.py
    """
    cohort_directory = r"Z:\debug_vids\Lynn_label_frames"
    
    # Ask for file type selection
    print("Select file type to clear:")
    print("1. Middle frame images (*_middle_frame.png)")
    print("2. ROI files (*_roi.json)")
    print("3. Brightness data files (*_brightness_data.csv)")
    
    try:
        selection = int(input("Enter selection (1-3): "))
        if selection == 1:
            file_type = 'images'
        elif selection == 2:
            file_type = 'roi'
        elif selection == 3:
            file_type = 'brightness'
        else:
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return
    
    # Ask if this should be a dry run (preview only)
    dry_run_input = input("Dry run (preview only, no deletion)? (y/n): ")
    dry_run = dry_run_input.lower() == 'y'
    
    # Ask for confirmation before proceeding with real deletion
    if not dry_run:
        print(f"WARNING: This will delete all {file_type} from {cohort_directory}")
        confirmation = input("Are you sure you want to proceed? (y/n): ")
        
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Run the file clearing function
    clear_files(cohort_directory, file_type, dry_run)

if __name__ == "__main__":
    main()