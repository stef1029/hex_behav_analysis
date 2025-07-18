#!/usr/bin/env python3
"""
Script to rename files and folders by replacing mouse ID in experimental session data.

This script identifies the current mouse ID from the session folder name and replaces
it with a new mouse ID throughout all filenames and the folder name itself.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Tuple, Optional


def extract_mouse_id_from_folder(folder_name: str) -> Optional[str]:
    """
    Extract the mouse ID from a session folder name.
    
    Expected format: YYMMDD_HHMMSS_mouseID
    
    Args:
        folder_name: Name of the session folder
        
    Returns:
        The mouse ID if found, None otherwise
    """
    # Pattern matches date_time_mouseID format
    pattern = r'^\d{6}_\d{6}_(.+)$'
    match = re.match(pattern, folder_name)
    
    if match:
        return match.group(1)
    return None


def rename_file_with_new_mouse_id(filepath: Path, old_mouse_id: str, new_mouse_id: str) -> Path:
    """
    Rename a file by replacing the old mouse ID with the new one.
    
    Args:
        filepath: Path to the file to rename
        old_mouse_id: Current mouse ID to replace
        new_mouse_id: New mouse ID to use
        
    Returns:
        Path to the renamed file
    """
    filename = filepath.name
    new_filename = filename.replace(old_mouse_id, new_mouse_id)
    
    if filename != new_filename:
        new_filepath = filepath.parent / new_filename
        filepath.rename(new_filepath)
        return new_filepath
    
    return filepath


def rename_session_folder(folder_path: Path, old_mouse_id: str, new_mouse_id: str) -> Path:
    """
    Rename the session folder by replacing the mouse ID.
    
    Args:
        folder_path: Path to the session folder
        old_mouse_id: Current mouse ID to replace
        new_mouse_id: New mouse ID to use
        
    Returns:
        Path to the renamed folder
    """
    folder_name = folder_path.name
    new_folder_name = folder_name.replace(old_mouse_id, new_mouse_id)
    
    if folder_name != new_folder_name:
        new_folder_path = folder_path.parent / new_folder_name
        folder_path.rename(new_folder_path)
        return new_folder_path
    
    return folder_path


def process_session_folder(session_folder_path: str, new_mouse_id: str) -> None:
    """
    Process a session folder to rename all files and the folder itself with a new mouse ID.
    
    Args:
        session_folder_path: Path to the session folder
        new_mouse_id: New mouse ID to use for renaming
    """
    # Convert to Path object
    folder_path = Path(session_folder_path)
    
    # Verify the folder exists
    if not folder_path.exists():
        raise FileNotFoundError(f"Session folder not found: {session_folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {session_folder_path}")
    
    # Extract current mouse ID from folder name
    folder_name = folder_path.name
    current_mouse_id = extract_mouse_id_from_folder(folder_name)
    
    if not current_mouse_id:
        raise ValueError(f"Could not extract mouse ID from folder name: {folder_name}")
    
    print(f"Current mouse ID: {current_mouse_id}")
    print(f"New mouse ID: {new_mouse_id}")
    
    if current_mouse_id == new_mouse_id:
        print("Current and new mouse IDs are the same. No changes needed.")
        return
    
    # First, rename all files in the folder
    print("\nRenaming files...")
    files_renamed = 0
    
    for filepath in folder_path.iterdir():
        if filepath.is_file():
            old_name = filepath.name
            new_filepath = rename_file_with_new_mouse_id(filepath, current_mouse_id, new_mouse_id)
            if old_name != new_filepath.name:
                print(f"  {old_name} -> {new_filepath.name}")
                files_renamed += 1
    
    print(f"\nRenamed {files_renamed} files")
    
    # Then rename the folder itself
    print("\nRenaming folder...")
    old_folder_name = folder_path.name
    new_folder_path = rename_session_folder(folder_path, current_mouse_id, new_mouse_id)
    print(f"  {old_folder_name} -> {new_folder_path.name}")
    
    print("\nRenaming complete!")


def main():
    """
    Main function to execute the mouse ID renaming process.
    """
    cohort_directory = Path("/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Training")
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder

    cohort = Cohort_folder(cohort_directory, OEAB_legacy=False, use_existing_cohort_info=False)

    session_list = [
   "250401_170041_mtao108-1e",
   "250402_170756_mtao108-1e",
   "250402_171438_mtao108-1e",
   "250404_164232_mtao108-1e",
   "250408_203246_mtao108-1e",
   "250408_203353_mtao108-1e",
   "250408_203544_mtao108-1e"
]
    for session in session_list:
        session_folder_path = cohort.get_session(session)['directory']
        new_mouse_id = 'mtao108-3e'  # Replace with the actual new mouse ID you want to use


        try:
            process_session_folder(session_folder_path, new_mouse_id)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return 0


if __name__ == "__main__":
    exit(main())