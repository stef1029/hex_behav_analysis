#!/usr/bin/env python3
"""
Ephys File Renamer Script

This script uses the Cohort_folder class to find ephys files and rename .bin files
to match the naming convention of corresponding .set files.

For example:
- If we have "ephys.set" and "data.bin" in the same folder
- The script will rename "data.bin" to "ephys.bin"

Author: Assistant
Date: 2025
"""

import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder

# Import the Cohort_folder class from the existing module
# Note: Adjust this import path based on your actual module structure
# from hex_behav_analysis.cohort_folder import Cohort_folder

# For this script, we'll use the provided Cohort_folder class directly
# [The full Cohort_folder class would be imported here in practice]


def find_bin_and_set_pairs(ephys_files: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Find corresponding .bin and .set files in the ephys files dictionary.
    
    Args:
        ephys_files: Dictionary of ephys files organised by extension
        
    Returns:
        Tuple of (bin_file_path, set_file_path) if both found, None otherwise
    """
    bin_file = ephys_files.get('bin')
    set_file = ephys_files.get('set')
    
    if bin_file and set_file:
        return bin_file, set_file
    return None


def extract_base_name_from_set_file(set_file_path: str) -> str:
    """
    Extract the base name from a .set file path (everything before the .set extension).
    
    Args:
        set_file_path: Full path to the .set file
        
    Returns:
        Base name without extension
    """
    set_path = Path(set_file_path)
    return set_path.stem  # This gets the filename without the extension


def rename_bin_file(bin_file_path: str, new_base_name: str) -> bool:
    """
    Rename a .bin file to use the new base name while preserving the .bin extension.
    
    Args:
        bin_file_path: Current path to the .bin file
        new_base_name: New base name to use
        
    Returns:
        True if rename was successful, False otherwise
    """
    try:
        bin_path = Path(bin_file_path)
        parent_dir = bin_path.parent
        new_bin_path = parent_dir / f"{new_base_name}.bin"
        
        # Check if the target file already exists
        if new_bin_path.exists():
            print(f"  WARNING: Target file already exists: {new_bin_path}")
            return False
        
        # Perform the rename
        bin_path.rename(new_bin_path)
        print(f"  ✓ Renamed: {bin_path.name} → {new_bin_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error renaming {bin_file_path}: {str(e)}")
        return False


def process_cohort_ephys_files(cohort_directory: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Process all ephys files in a cohort directory and rename .bin files to match .set files.
    
    Args:
        cohort_directory: Path to the cohort directory
        dry_run: If True, only simulate the renaming without actually changing files
        
    Returns:
        Dictionary with statistics about the renaming process
    """
    print(f"Processing cohort directory: {cohort_directory}")
    print(f"Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 60)
    
    # Initialise the cohort folder with ephys data scanning
    try:
        cohort = Cohort_folder(
            cohort_directory,
            multi=True,
            portable_data=False,
            OEAB_legacy=False,
            ignore_tests=True,
            ephys_data=True
        )
    except Exception as e:
        print(f"Error initialising Cohort_folder: {str(e)}")
        return {"error": 1}
    
    # Statistics tracking
    stats = {
        "sessions_processed": 0,
        "sessions_with_ephys": 0,
        "bin_files_found": 0,
        "set_files_found": 0,
        "successful_renames": 0,
        "failed_renames": 0,
        "skipped_no_set": 0,
        "skipped_already_named": 0
    }
    
    # Process each session
    for mouse_id in cohort.cohort["mice"]:
        for session_id in cohort.cohort["mice"][mouse_id]["sessions"]:
            stats["sessions_processed"] += 1
            session = cohort.cohort["mice"][mouse_id]["sessions"][session_id]
            
            # Check if this session has ephys data
            if "ephys_data" not in session:
                continue
                
            ephys_files = session["ephys_data"]
            
            # Skip if no ephys files found
            if not ephys_files.get("has_ephys_data", False):
                continue
                
            stats["sessions_with_ephys"] += 1
            print(f"\nProcessing session: {session_id} (Mouse: {mouse_id})")
            
            # Look for .bin and .set file pairs
            file_pair = find_bin_and_set_pairs(ephys_files)
            
            if not file_pair:
                # Check what files we do have
                if 'bin' in ephys_files:
                    stats["bin_files_found"] += 1
                    stats["skipped_no_set"] += 1
                    print(f"  - Found .bin file but no corresponding .set file")
                if 'set' in ephys_files:
                    stats["set_files_found"] += 1
                continue
            
            bin_file_path, set_file_path = file_pair
            stats["bin_files_found"] += 1
            stats["set_files_found"] += 1
            
            # Extract base name from .set file
            set_base_name = extract_base_name_from_set_file(set_file_path)
            bin_base_name = extract_base_name_from_set_file(bin_file_path)
            
            print(f"  - Found .bin file: {Path(bin_file_path).name}")
            print(f"  - Found .set file: {Path(set_file_path).name}")
            print(f"  - Set base name: '{set_base_name}'")
            print(f"  - Bin base name: '{bin_base_name}'")
            
            # Check if the .bin file already has the correct name
            if bin_base_name == set_base_name:
                stats["skipped_already_named"] += 1
                print(f"  - .bin file already has correct name, skipping")
                continue
            
            # Perform the rename (or simulate it)
            if dry_run:
                new_bin_name = f"{set_base_name}.bin"
                print(f"  - [DRY RUN] Would rename: {Path(bin_file_path).name} → {new_bin_name}")
                stats["successful_renames"] += 1
            else:
                success = rename_bin_file(bin_file_path, set_base_name)
                if success:
                    stats["successful_renames"] += 1
                else:
                    stats["failed_renames"] += 1
    
    return stats


def print_summary_statistics(stats: Dict[str, int]) -> None:
    """
    Print a summary of the renaming process statistics.
    
    Args:
        stats: Dictionary containing process statistics
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Sessions processed: {stats['sessions_processed']}")
    print(f"Sessions with ephys data: {stats['sessions_with_ephys']}")
    print(f"Bin files found: {stats['bin_files_found']}")
    print(f"Set files found: {stats['set_files_found']}")
    print(f"Successful renames: {stats['successful_renames']}")
    print(f"Failed renames: {stats['failed_renames']}")
    print(f"Skipped (no .set file): {stats['skipped_no_set']}")
    print(f"Skipped (already correct name): {stats['skipped_already_named']}")
    print("=" * 60)


def main():
    """
    Main function to execute the ephys file renaming process.
    """
    # Configuration
    cohort_directory = Path(r"Z://Behaviour/2504_pitx_ephys_cohort")
    
    # Set to True for a dry run (simulation without actual file changes)
    dry_run = False
    
    print("Ephys File Renamer")
    print("=" * 60)
    
    # Validate cohort directory exists
    if not cohort_directory.exists():
        print(f"Error: Cohort directory does not exist: {cohort_directory}")
        return
    
    # Process the cohort
    try:
        stats = process_cohort_ephys_files(cohort_directory, dry_run=dry_run)
        
        # Check for errors
        if "error" in stats:
            print("Failed to process cohort due to errors.")
            return
        
        # Print summary
        print_summary_statistics(stats)
        
        if dry_run:
            print("\nThis was a DRY RUN. No files were actually renamed.")
            print("Set dry_run=False to perform actual renaming.")
        else:
            print(f"\nFile renaming completed!")
            
    except Exception as e:
        print(f"Unexpected error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()