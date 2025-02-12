from pathlib import Path
import subprocess
from Cohort_folder import Cohort_folder
from Session_nwb import Session
from pynwb import NWBHDF5IO
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def create_cohort_folder(source):
    cohort = Cohort_folder(source, multi=True)
    return cohort

def check_and_process_nwb_file(nwb_file_path, cohort):
    with NWBHDF5IO(str(nwb_file_path), 'r+') as io:  # Open in read/write mode
        nwbfile = io.read()
        
        # Check if 'behaviour_coords' is present
        if 'behaviour_coords' in nwbfile.processing:
            print(f"'behaviour_coords' already present in {nwb_file_path.name}.")
        else:
            print(f"'behaviour_coords' not present in {nwb_file_path.name}. Creating Session object.")
            # Create a session instance which will automatically add the behaviour_coords
            session_name = nwb_file_path.stem 
            session_instance = Session(cohort.get_session(session_name))
            session_instance  # Instantiating the session will add the coordinates

def process_and_rsync_files(src: Path, dst: Path, cohort: Cohort_folder):
    # Get a list of all .nwb files and the specific cohort_info.png to process
    files_to_process = [
        item for item in src.rglob('*')
        if item.is_file() and (
            item.suffix == '.nwb' or (item.suffix == '.png' and item.name == 'cohort_info.png')
        ) and 'OEAB' not in item.parts
    ]
    total_files = len(files_to_process)
    
    for count, item in enumerate(files_to_process, start=1):
        # Process the nwb file
        if item.suffix == '.nwb':
            check_and_process_nwb_file(item, cohort)
        
        # Calculate the relative destination path
        relative_path = item.relative_to(src)
        destination_path = dst / relative_path

        # Ensure the parent directory exists in the destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rsync to copy the file
        rsync_command = ['rsync', '-av', '--progress', str(item), str(destination_path)]
        subprocess.run(rsync_command)
        
        # Display progress with color
        print(Fore.GREEN + f"Processed and copied {count}/{total_files}: {item.name}" + Style.RESET_ALL)

def main():

    # # Set your source and destination directories here
    # source_directory = Path(r'/cephfs2/srogers/Behaviour code/2407_July_WT_cohort/Data')
    # destination_directory = Path(r'/cephfs2/srogers/Behaviour code/2407_July_WT_cohort/Portable_data')

    # Set your source and destination directories here
    # # source_directory = Path(r'/cephfs2/srogers/March_training')
    # # destination_directory = Path(r'/cephfs2/srogers/Behaviour code/March_training_portable')
    
    # # # Initialize cohort
    # # cohort = create_cohort_folder(source_directory)

    # # # Execute the processing and rsync process
    # # process_and_rsync_files(source_directory, destination_directory, cohort)

    # # Set your source and destination directories here
    source_directory = Path(r'/cephfs2/srogers/Behaviour code/2409_September_cohort/DATA_ArduinoDAQ')
    destination_directory = Path(r'/cephfs2/srogers/Behaviour code/Portable_data/September_portable_AD')
    source_directory = Path(r'/cephfs2/dwelch/Behaviour/November_cohort')
    destination_directory = Path(r'/cephfs2/srogers/Behaviour code/Portable_data/Dan_december_cohort')

    # Initialize cohort
    cohort = create_cohort_folder(source_directory)

    # Execute the processing and rsync process
    process_and_rsync_files(source_directory, destination_directory, cohort)

if __name__ == "__main__":
    main()
