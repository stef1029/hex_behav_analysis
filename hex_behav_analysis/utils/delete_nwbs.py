import os
import sys
from pathlib import Path

# Import the Cohort_folder class assuming it's available in the same directory
try:
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
except ImportError:
    # If the import fails, add a helpful error message
    print("Error: Could not import Cohort_folder. Make sure the hex_behav_analysis package is installed.")
    sys.exit(1)


def delete_nwb_files(cohort, test_mode=True):
    """
    Delete or list all NWB files in a cohort.
    
    This function either lists (in test mode) or deletes all NWB files
    that are associated with sessions in the provided cohort.
    
    Parameters
    ----------
    cohort : Cohort_folder
        A Cohort_folder object containing the session information.
    test_mode : bool, default=True
        If True, only list files that would be deleted without removing them.
        If False, actually delete the NWB files.
        
    Returns
    -------
    dict
        A dictionary with statistics about the operation:
        - 'files_found': Total number of NWB files found
        - 'files_deleted': Number of files deleted (0 in test mode)
        - 'files_skipped': Number of files skipped (not found or "None")
        - 'total_size_mb': Total size of files in MB (deleted or would be deleted)
        - 'file_paths': List of file paths that were deleted or would be deleted
    """
    import os
    from pathlib import Path
    
    stats = {
        'files_found': 0,
        'files_deleted': 0,
        'files_skipped': 0,
        'total_size_mb': 0,
        'file_paths': []  # New list to store paths of files
    }
    
    # Action to perform based on mode
    action_text = "Would delete" if test_mode else "Deleting"
    
    print(f"{'TEST MODE: ' if test_mode else ''}Scanning for NWB files in cohort '{cohort.cohort_directory.name}'...")
    print("\nFiles that {0} be deleted:".format("would" if test_mode else "will"))
    print("-" * 80)
    
    # Track all NWB paths to handle potential duplicates
    all_nwb_paths = set()
    
    # First pass: Locate all NWB files
    for mouse in cohort.cohort["mice"]:
        for session_id in cohort.cohort["mice"][mouse]["sessions"]:
            session = cohort.cohort["mice"][mouse]["sessions"][session_id]
            
            # Try to get NWB file path from the session dictionary
            nwb_path = session.get("NWB_file")
            
            # If no explicit NWB_file field, check in processed_data
            if not nwb_path and "processed_data" in session:
                nwb_path = session["processed_data"].get("NWB_file")
            
            # Skip if no NWB file or placeholder "None" string
            if not nwb_path or nwb_path == "None":
                stats['files_skipped'] += 1
                continue
            
            # Convert to Path object for consistency
            nwb_path = Path(nwb_path)
            
            # Skip if already processed or doesn't exist
            if str(nwb_path) in all_nwb_paths or not nwb_path.exists():
                stats['files_skipped'] += 1
                continue
                
            # Add to set of found paths
            all_nwb_paths.add(str(nwb_path))
            stats['file_paths'].append(str(nwb_path))  # Store path in stats
            stats['files_found'] += 1
            
            # Get file size
            file_size_mb = nwb_path.stat().st_size / (1024 * 1024)  # Convert bytes to MB
            stats['total_size_mb'] += file_size_mb
            
            # Print information about the file
            mouse_session = f"{mouse}/{session_id}"
            print(f"{stats['files_found']:3d}. {action_text}: {nwb_path} ({file_size_mb:.2f} MB)")
            print(f"    Associated with: {mouse_session}")
    
    # If no files were found, print a message
    if stats['files_found'] == 0:
        print("No NWB files found.")
    
    # Print a divider
    print("-" * 80)
    
    # Second pass: Delete files if not in test mode
    if not test_mode:
        print("\nDeleting files:")
        print("-" * 80)
        for nwb_path in all_nwb_paths:
            try:
                os.remove(nwb_path)
                stats['files_deleted'] += 1
                print(f"Successfully deleted: {nwb_path}")
            except Exception as e:
                print(f"Error deleting {nwb_path}: {str(e)}")
                stats['files_deleted'] -= 1  # Decrement count for failed deletion
        print("-" * 80)
    
    # Print summary
    print("\nSummary:")
    print(f"Total NWB files found:       {stats['files_found']}")
    if not test_mode:
        print(f"Files successfully deleted:  {stats['files_deleted']}")
    print(f"Files skipped:              {stats['files_skipped']} (not found or 'None')")
    print(f"Total size:                 {stats['total_size_mb']:.2f} MB")
    
    if test_mode:
        print("\nThis was a TEST RUN. No files were actually deleted.")
        print("To perform actual deletion, set test_mode=False in the main function")
        
        # Save the list of files to a text file for reference
        save_path = cohort.cohort_directory / "nwb_files_to_delete.txt"
        with open(save_path, 'w') as f:
            f.write(f"NWB files that would be deleted from cohort: {cohort.cohort_directory.name}\n")
            f.write(f"Generated on: {os.path.basename(__file__)} at {Path(__file__).stat().st_mtime}\n\n")
            for i, path in enumerate(stats['file_paths'], 1):
                f.write(f"{i}. {path}\n")
        print(f"\nList of files to delete saved to: {save_path}")
    
    return stats

def main():
    """
    Main function to initialise a Cohort_folder object and delete NWB files.
    """
    # Define your variables here
    cohort_directory = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"  # Change this to your cohort directory
    
    # Cohort_folder parameters
    multi = True                    # Whether data is split across subfolders (multiple mice)
    portable_data = False            # Use portable_data logic vs. full raw-data logic
    oeab_legacy = False              # Look for legacy OEAB folder structures
    ignore_tests = True             # Skip any session folders that look like test sessions
    use_existing_cohort_info = False # If True and cohort_info.json exists, load from it
    plot = False                    # Whether to produce a cohort summary plot
    
    # Delete NWB files parameters
    test_mode = False                # Set to False to actually delete files
    
    # Verify that the directory exists
    cohort_dir = Path(cohort_directory)
    if not cohort_dir.exists() or not cohort_dir.is_dir():
        print(f"Error: Directory does not exist: {cohort_dir}")
        sys.exit(1)
    
    try:
        # Initialise Cohort_folder object with the specified parameters
        print(f"Initialising Cohort_folder for: {cohort_dir}")
        cohort = Cohort_folder(
            cohort_directory=cohort_dir,
            multi=multi,
            portable_data=portable_data,
            OEAB_legacy=oeab_legacy,
            ignore_tests=ignore_tests,
            use_existing_cohort_info=use_existing_cohort_info,
            plot=plot
        )
        
        # Call the delete_nwb_files function with the cohort object
        delete_nwb_files(cohort, test_mode=test_mode)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()