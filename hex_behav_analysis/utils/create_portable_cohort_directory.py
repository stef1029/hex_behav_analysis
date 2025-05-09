from pathlib import Path
import subprocess
from pynwb import NWBHDF5IO
from colorama import init, Fore, Style
import logging
import traceback
from datetime import datetime

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session

# Initialize colorama
init(autoreset=True)

# Set up logging
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'processing_errors_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

def create_cohort_folder(source):
    try:
        cohort = Cohort_folder(source, multi=True)
        return cohort
    except Exception as e:
        logging.error(f"Failed to create cohort folder for {source}:\n{str(e)}\n{traceback.format_exc()}")
        return None

def check_and_process_nwb_file(nwb_file_path, cohort):
    try:
        with NWBHDF5IO(str(nwb_file_path), 'r+') as io:
            nwbfile = io.read()
            
            if 'behaviour_coords' in nwbfile.processing:
                logging.info(f"'behaviour_coords' already present in {nwb_file_path.name}.")
            else:
                logging.info(f"'behaviour_coords' not present in {nwb_file_path.name}. Creating Session object.")
                session_name = nwb_file_path.stem 
                session_instance = Session(cohort.get_session(session_name))
                session_instance  # Instantiating the session will add the coordinates
        return True
    except Exception as e:
        logging.error(f"Error processing NWB file {nwb_file_path}:\n{str(e)}\n{traceback.format_exc()}")
        return False

def rsync_file(source_path, dest_path):
    try:
        rsync_command = ['rsync', '-av', '--progress', str(source_path), str(dest_path)]
        result = subprocess.run(rsync_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"rsync failed with error:\n{result.stderr}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to rsync {source_path} to {dest_path}:\n{str(e)}\n{traceback.format_exc()}")
        return False

def process_and_rsync_files(src: Path, dst: Path, cohort: Cohort_folder, selected_sessions=None):
    if not cohort:
        logging.error("No valid cohort object provided. Aborting processing.")
        return

    try:
        # Get all eligible files first
        all_files = [
            item for item in src.rglob('*')
            if item.is_file() and (
                item.suffix == '.nwb' or (item.suffix == '.png' and item.name == 'cohort_info.png')
            ) and 'OEAB' not in item.parts
        ]
        
        # Filter by session ID if specified
        if selected_sessions:
            files_to_process = []
            for item in all_files:
                # Always include cohort_info.png
                if item.name == 'cohort_info.png':
                    files_to_process.append(item)
                    continue
                    
                # For NWB files, check if they belong to a selected session
                if item.suffix == '.nwb':
                    session_name = item.stem  # This assumes the stem (filename without extension) is the session ID
                    # Check if this session is in our selected list
                    if any(session_id in session_name for session_id in selected_sessions):
                        files_to_process.append(item)
        else:
            # If no sessions specified, process all files
            files_to_process = all_files
            
        total_files = len(files_to_process)
        
        if total_files == 0:
            logging.warning(f"No files found to process in {src}")
            return

        successful_files = 0
        failed_files = 0

        for count, item in enumerate(files_to_process, start=1):
            try:
                relative_path = item.relative_to(src)
                destination_path = dst / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                process_success = True
                if item.suffix == '.nwb':
                    process_success = check_and_process_nwb_file(item, cohort)

                if process_success and rsync_file(item, destination_path):
                    successful_files += 1
                    print(Fore.GREEN + f"Processed and copied {count}/{total_files}: {item.name}" + Style.RESET_ALL)
                else:
                    failed_files += 1
                    print(Fore.RED + f"Failed to process/copy {count}/{total_files}: {item.name}" + Style.RESET_ALL)

            except Exception as e:
                failed_files += 1
                logging.error(f"Error processing file {item}:\n{str(e)}\n{traceback.format_exc()}")
                print(Fore.RED + f"Error processing {count}/{total_files}: {item.name}" + Style.RESET_ALL)
                continue  # Continue with next file

        # Log summary
        logging.info(f"\nProcessing Summary:")
        logging.info(f"Total files: {total_files}")
        logging.info(f"Successfully processed: {successful_files}")
        logging.info(f"Failed: {failed_files}")

    except Exception as e:
        logging.error(f"Fatal error in process_and_rsync_files:\n{str(e)}\n{traceback.format_exc()}")

def main():
    try:
        # Set up logging
        log_file = setup_logging()
        logging.info("Starting processing script")
        
        # Set your source and destination directories here
        source_directory = Path(r'/cephfs2/srogers/Behaviour code/2409_September_cohort/DATA_ArduinoDAQ')
        # source_directory = Path(r'/cephfs2/dwelch/Behaviour/November_cohort')
        destination_directory = Path(r'/cephfs2/srogers/Behaviour code/Portable_data/sep')

        # Specify the session IDs you want to process
        # Leave as None to process all sessions, or specify a list of session IDs
        # Example: selected_sessions = ['session1', 'session2', 'session3']
        selected_sessions = []  # Change this to a list of sessions to process specific sessions
        
        # Log the directories being used
        logging.info(f"Source directory: {source_directory}")
        logging.info(f"Destination directory: {destination_directory}")
        
        # Log specified sessions if any
        if selected_sessions:
            logging.info(f"Processing only these sessions: {selected_sessions}")
        else:
            logging.info("Processing all sessions (no filtering)")

        # Initialize cohort
        logging.info("Initializing cohort folder")
        cohort = create_cohort_folder(source_directory)

        # Execute the processing and rsync process
        logging.info("Starting file processing and transfer")
        process_and_rsync_files(source_directory, destination_directory, cohort, selected_sessions)

        logging.info("Script completed. Check the log file for details: " + str(log_file))
        print(Fore.GREEN + f"Processing complete. Log file: {log_file}" + Style.RESET_ALL)

    except Exception as e:
        logging.error(f"Fatal error in main function:\n{str(e)}\n{traceback.format_exc()}")
        print(Fore.RED + "Fatal error occurred. Check the log file for details." + Style.RESET_ALL)

if __name__ == "__main__":
    main()