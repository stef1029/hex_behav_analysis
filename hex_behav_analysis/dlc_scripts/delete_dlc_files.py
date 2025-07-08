#!/usr/bin/env python3
"""
Script to delete all DLC-related files from a cohort folder.
This includes:
- CSV coordinate files
- H5 coordinate files
- Pickle files (both full and meta)
- Labelled videos
- Split files
"""

import argparse
from pathlib import Path
import re
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder


def is_dlc_file(filename):
    """
    Check if a file is a DLC-related file.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if this is a DLC file
    """
    filename_lower = filename.lower()
    
    # Check for labelled/labeled videos
    if 'labeled' in filename_lower or 'labelled' in filename_lower:
        return True
    
    # Check for DLC in filename (CSV, H5, pickle files)
    if 'dlc' in filename_lower:
        return True
    
    # Check for split files
    if 'split' in filename_lower:
        return True
    
    # Check for ResNet models (in case DLC isn't in the filename)
    if 'resnet' in filename_lower:
        return True
    
    return False


def find_dlc_files_in_session(session_folder):
    """
    Find all DLC-related files in a session folder.
    
    Args:
        session_folder (Path): Path to the session folder
        
    Returns:
        list: List of Path objects for DLC files found
    """
    dlc_files = []
    
    # Convert to Path object
    session_path = Path(session_folder)
    
    # Find all files in the session folder
    for file_path in session_path.glob("*"):
        if file_path.is_file() and is_dlc_file(file_path.name):
            dlc_files.append(file_path)
    
    return dlc_files


def delete_dlc_files_from_cohort(cohort_directory, multi=True, portable_data=False, 
                                 dry_run=True, verbose=True):
    """
    Delete all DLC files from a cohort.
    
    Args:
        cohort_directory (str or Path): Path to the cohort directory
        multi (bool): Whether the data is split across subfolders (multiple mice)
        portable_data (bool): Whether to use the 'portable_data' logic
        dry_run (bool): If True, only show what would be deleted without actually deleting
        verbose (bool): If True, print detailed information
        
    Returns:
        dict: Summary of files deleted/found organised by mouse and session
    """
    print(f"{'DRY RUN - ' if dry_run else ''}Scanning cohort for DLC files...")
    
    # Load cohort information
    cohort = Cohort_folder(
        cohort_directory,
        multi=multi,
        portable_data=portable_data,
        use_existing_cohort_info=True
    )
    
    deletion_summary = {}
    total_files_found = 0
    total_size_mb = 0
    
    # Iterate through all mice and sessions
    for mouse in cohort.cohort["mice"]:
        deletion_summary[mouse] = {}
        
        for session in cohort.cohort["mice"][mouse]["sessions"]:
            session_folder = Path(cohort.cohort["mice"][mouse]["sessions"][session]["directory"])
            
            # Find all DLC files in this session
            dlc_files = find_dlc_files_in_session(session_folder)
            
            if dlc_files:
                deletion_summary[mouse][session] = []
                
                for file_path in dlc_files:
                    file_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    deletion_summary[mouse][session].append(file_info)
                    total_files_found += 1
                    total_size_mb += file_info['size_mb']
                    
                    if verbose:
                        print(f"  Found: {file_path.name} ({file_info['size_mb']:.2f} MB)")
                    
                    if not dry_run:
                        try:
                            file_path.unlink()
                            if verbose:
                                print(f"    ✓ Deleted")
                        except Exception as e:
                            print(f"    ✗ Error deleting {file_path}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"{'DRY RUN ' if dry_run else ''}SUMMARY")
    print("="*50)
    print(f"Total files found: {total_files_found}")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    if dry_run:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("Run with --no-dry-run to actually delete files.")
    else:
        print(f"\n✓ Deleted {total_files_found} files")
    
    return deletion_summary


def main():
    """
    Main function to handle command line arguments and execute deletion.
    """
    parser = argparse.ArgumentParser(
        description='Delete all DLC-related files from a cohort folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - just show what would be deleted
  python delete_dlc_files.py /path/to/cohort
  
  # Actually delete the files
  python delete_dlc_files.py /path/to/cohort --no-dry-run
  
  # For portable data format
  python delete_dlc_files.py /path/to/cohort --portable-data
  
  # For single mouse data (not multi-folder structure)
  python delete_dlc_files.py /path/to/cohort --no-multi
        """
    )
    
    parser.add_argument('cohort_directory', type=str,
                        help='Path to the cohort directory')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually delete files (default is dry run)')
    parser.add_argument('--no-multi', dest='multi', action='store_false',
                        help='Data is not split across subfolders')
    parser.add_argument('--portable-data', action='store_true',
                        help='Use portable data format')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Convert cohort directory to Path
    cohort_dir = Path(args.cohort_directory)
    if not cohort_dir.exists():
        print(f"Error: Cohort directory '{cohort_dir}' does not exist")
        return 1
    
    # Run the deletion
    try:
        delete_dlc_files_from_cohort(
            cohort_directory=cohort_dir,
            multi=args.multi,
            portable_data=args.portable_data,
            dry_run=not args.no_dry_run,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())