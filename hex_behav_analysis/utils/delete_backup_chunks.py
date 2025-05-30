import os
import glob
import re

def clean_backup_files(root_dir):
    """
    Removes individual .npy files from backup_files folders when a complete backup file exists.
    
    Scans through an electrophysiology data directory structure to find session folders with
    complete backup files, and deletes the individual .npy files in the corresponding 
    backup_files folders.
    
    Args:
        root_dir (str): Path to the root directory
    
    Returns:
        tuple: (int, int) - Count of deleted files and count of skipped folders
    """
    deleted_count = 0
    skipped_folders = 0
    
    # Find all date_time folders
    date_time_folders = [f for f in glob.glob(os.path.join(root_dir, "*")) 
                        if os.path.isdir(f) and re.match(r"\d{6}_\d{6}", os.path.basename(f))]
    
    print(f"Found {len(date_time_folders)} date/time folders")
    
    for date_time_folder in date_time_folders:
        date_time_name = os.path.basename(date_time_folder)
        print(f"Processing: {date_time_name}")
        
        # Find all session folders within this date_time folder
        session_folders = [f for f in glob.glob(os.path.join(date_time_folder, "*")) 
                          if os.path.isdir(f)]
        
        for session_folder in session_folders:
            session_name = os.path.basename(session_folder)
            
            # Check for backup_files folder
            backup_files_folder = os.path.join(session_folder, "backup_files")
            
            if not os.path.exists(backup_files_folder):
                continue
                
            # Check if the complete backup file exists in the same folder as the session folder
            complete_backup_file = os.path.join(session_folder, f"{session_name}-complete-backup.npy")
            
            if os.path.exists(complete_backup_file):
                # Get list of .npy files in the backup_files folder
                npy_files = glob.glob(os.path.join(backup_files_folder, "*.npy"))
                
                if npy_files:
                    print(f"  Found complete backup for {session_name}. Deleting {len(npy_files)} .npy files.")
                    
                    for npy_file in npy_files:
                        os.remove(npy_file)
                        deleted_count += 1
            else:
                skipped_folders += 1
    
    return deleted_count, skipped_folders

def main():
    """
    Main function to run the cleanup process.
    """
    # Set the root directory - modify this to your actual path
    root_directory = "E:\\2504_pitx_ephys_cohort"
    
    print(f"Starting cleanup process in {root_directory}")
    print("This script will delete .npy files in backup_files folders where a complete backup exists.")
    
    try:
        # Add a safety prompt
        confirmation = input("Do you want to proceed? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return
        
        deleted, skipped = clean_backup_files(root_directory)
        
        print(f"\nCleanup complete!")
        print(f"Deleted {deleted} individual .npy files")
        print(f"Skipped {skipped} folders without complete backups")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

if __name__ == "__main__":
    main()