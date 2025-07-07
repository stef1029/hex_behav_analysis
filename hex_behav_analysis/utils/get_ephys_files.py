#!/usr/bin/env python3
"""
Script to calculate total storage size of ephys files in a behaviour cohort directory.
Uses the Cohort_folder class to find ephys files and calculates their combined size.
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import humanize

# Import your Cohort_folder class
# Adjust this import based on your actual module structure
from Cohort_folder import Cohort_folder


def calculate_file_size(file_path):
    """
    Calculate the size of a file in bytes.
    
    :param file_path: Path to the file
    :return: Size in bytes, or 0 if file doesn't exist
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.stat().st_size
    except Exception as e:
        print(f"Warning: Could not get size of {file_path}: {e}")
    return 0


def calculate_ephys_storage(cohort_directory, multi=True, use_existing=True):
    """
    Calculate total storage size of all ephys files in a cohort directory.
    
    :param cohort_directory: Path to the cohort directory
    :param multi: Whether the data is split across subfolders
    :param use_existing: Whether to use existing cohort_info.json if available
    :return: Total size in bytes
    """
    print(f"Loading cohort information from: {cohort_directory}")
    print("This may take a moment if scanning for the first time...")
    
    # Create cohort folder instance with ephys data scanning enabled
    cohort = Cohort_folder(
        cohort_directory,
        multi=multi,
        portable_data=False,  # Set to False for raw data with ephys
        use_existing_cohort_info=use_existing,
        plot=False,
        ephys_data=True  # Enable ephys file scanning
    )
    
    # Count total sessions for progress bar
    total_sessions = sum(
        len(cohort.cohort["mice"][mouse]["sessions"]) 
        for mouse in cohort.cohort["mice"]
    )
    
    print(f"\nFound {len(cohort.cohort['mice'])} mice with {total_sessions} total sessions")
    print("Calculating ephys file sizes...\n")
    
    # Variables to track storage
    total_size = 0
    file_count = 0
    session_with_ephys = 0
    size_by_extension = {}
    
    # Create progress bar
    with tqdm(total=total_sessions, desc="Processing sessions", unit="session") as pbar:
        for mouse in cohort.cohort["mice"]:
            mouse_data = cohort.cohort["mice"][mouse]
            
            for session_id in mouse_data["sessions"]:
                session = mouse_data["sessions"][session_id]
                
                # Check if session has ephys data
                if "ephys_data" in session:
                    ephys_files = session["ephys_data"]
                    
                    # Check if there are actually ephys files
                    if ephys_files.get("has_ephys_data", False):
                        session_with_ephys += 1
                        
                        # Process each ephys file
                        for ext, file_path in ephys_files.items():
                            # Skip metadata fields
                            if ext in ['total_files', 'has_ephys_data', 'probe_file']:
                                continue
                            
                            if file_path and file_path != "None":
                                size = calculate_file_size(file_path)
                                if size > 0:
                                    total_size += size
                                    file_count += 1
                                    
                                    # Track size by extension
                                    if ext not in size_by_extension:
                                        size_by_extension[ext] = {'count': 0, 'size': 0}
                                    size_by_extension[ext]['count'] += 1
                                    size_by_extension[ext]['size'] += size
                        
                        # Also check probe file
                        probe_file = ephys_files.get("probe_file")
                        if probe_file and probe_file != "None":
                            size = calculate_file_size(probe_file)
                            if size > 0:
                                total_size += size
                                file_count += 1
                                if 'probe_file' not in size_by_extension:
                                    size_by_extension['probe_file'] = {'count': 0, 'size': 0}
                                size_by_extension['probe_file']['count'] += 1
                                size_by_extension['probe_file']['size'] += size
                
                # Update progress bar
                pbar.update(1)
    
    # Print results
    print("\n" + "="*60)
    print("EPHYS STORAGE SUMMARY")
    print("="*60)
    print(f"Cohort directory: {cohort_directory}")
    print(f"Total mice: {len(cohort.cohort['mice'])}")
    print(f"Total sessions: {total_sessions}")
    print(f"Sessions with ephys data: {session_with_ephys}")
    print(f"Total ephys files found: {file_count}")
    print(f"\nTotal storage size: {humanize.naturalsize(total_size, binary=True)}")
    print(f"                    ({total_size:,} bytes)")
    
    # Print breakdown by file type
    if size_by_extension:
        print("\n" + "-"*60)
        print("BREAKDOWN BY FILE TYPE")
        print("-"*60)
        print(f"{'Extension':<15} {'Count':>10} {'Size':>20}")
        print("-"*60)
        
        # Sort by size descending
        sorted_extensions = sorted(
            size_by_extension.items(), 
            key=lambda x: x[1]['size'], 
            reverse=True
        )
        
        for ext, data in sorted_extensions:
            size_str = humanize.naturalsize(data['size'], binary=True)
            print(f"{ext:<15} {data['count']:>10} {size_str:>20}")
    
    print("="*60)
    
    return total_size


def main():
    """Main function to run the ephys storage calculator."""

    cohort_directory = r"/cephfs2/srogers/Behaviour/2504_pitx_ephys_cohort"
    
    # Convert to Path object
    cohort_path = Path(cohort_directory)
    
    # Check if directory exists
    if not cohort_path.exists():
        print(f"Error: Directory {cohort_path} does not exist!")
        sys.exit(1)
    
    if not cohort_path.is_dir():
        print(f"Error: {cohort_path} is not a directory!")
        sys.exit(1)
    
    # Calculate storage
    try:
        total_size = calculate_ephys_storage(
            cohort_path,
            multi=True,
            use_existing=False
        )
    except Exception as e:
        print(f"\nError occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()