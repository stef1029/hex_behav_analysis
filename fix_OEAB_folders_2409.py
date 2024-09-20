import os
import shutil
from pathlib import Path

def check_and_copy_openephys(source_adc_openephys_file, source_full_openephys_file, base_directory):
    """
    This function checks whether the folder has only 122 ADC files or both 122 ADC and CH files.
    It copies the appropriate version of the .openephys file based on the files detected,
    ensuring that the source and target paths are not the same.
    Prints a list of folders that contain non-122 files and skips those folders.
    
    :param source_adc_openephys_file: Path to the ADC-only version of the structure.openephys file
    :param source_full_openephys_file: Path to the full ADC + CH version of the structure.openephys file
    :param base_directory: The root directory containing the session folders
    """
    non_122_folders = []  # To store folders that contain non-122 files

    # Iterate over all folders in the base directory
    for session_folder in Path(base_directory).glob('*'):
        if session_folder.is_dir():
            # Check for an OEAB folder in the session folder
            oeab_folder = next(session_folder.glob('*OEAB_recording*'), None)
            if oeab_folder and oeab_folder.is_dir():
                # Look into subfolders (like 'Record Node 120') inside the OEAB folder
                for subfolder in oeab_folder.glob('*'):
                    if subfolder.is_dir():
                        print(f"\nChecking subfolder: {subfolder}")
                        
                        # Check if there are any files with '122' in their names
                        has_122_files = any('122' in file.stem for file in subfolder.glob('*'))
                        
                        if has_122_files:
                            # Check for ADC and CH files
                            has_adc_files = any('ADC' in file.stem for file in subfolder.glob('*'))
                            has_ch_files = any('CH' in file.stem for file in subfolder.glob('*'))

                            # Determine the target file path
                            target_file = subfolder / "structure.openephys"
                            
                            # If the folder contains ADC files but no CH files, copy the ADC-only version
                            if has_adc_files and not has_ch_files:
                                # Only copy if the source and target are different
                                if source_adc_openephys_file != target_file:
                                    shutil.copy2(source_adc_openephys_file, target_file)
                                    print(f"Copied ADC-only version to {target_file}")
                                else:
                                    print(f"Skipping copy: source and target paths are the same for {target_file}")
                            
                            # If the folder contains both ADC and CH files, copy the full version
                            elif has_adc_files and has_ch_files:
                                # Only copy if the source and target are different
                                if source_full_openephys_file != target_file:
                                    shutil.copy2(source_full_openephys_file, target_file)
                                    print(f"Copied full ADC + CH version to {target_file}")
                                else:
                                    print(f"Skipping copy: source and target paths are the same for {target_file}")
                        else:
                            # If no 122 files are found, store this folder in the list
                            non_122_folders.append(subfolder)
                            print(f"Skipping {subfolder}, no 122 files found.")
            else:
                print(f"No OEAB folder found in {session_folder}")
    
    # Print the list of folders that contain non-122 files
    if non_122_folders:
        print("\nFolders that contain non-122 files:")
        for folder in non_122_folders:
            print(f"- {folder}")

# Example usage:
source_openephys_file = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/Data/240904_144526/240904_144526_OEAB_recording/Record Node 120/structure.openephys")
source_full_openephys_file = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/Data/structure.openephys")
base_directory = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/Data")

# Run the function
check_and_copy_openephys(source_openephys_file, source_full_openephys_file, base_directory)
