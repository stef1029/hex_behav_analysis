from pathlib import Path
import subprocess
from pynwb import NWBHDF5IO
from colorama import init, Fore, Style
import logging
import traceback
from datetime import datetime

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

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

def process_and_rsync_files(src: Path, dst: Path, cohort: Cohort_folder):
    if not cohort:
        logging.error("No valid cohort object provided. Aborting processing.")
        return

    try:
        files_to_process = [
            item for item in src.rglob('*')
            if item.is_file() and (
                item.suffix == '.nwb' or (item.suffix == '.png' and item.name == 'cohort_info.png')
            ) and 'OEAB' not in item.parts
        ]
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
        source_directory = Path(r'/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE')
        destination_directory = Path(r'/cephfs2/srogers/Behaviour code/Portable_data/2501_Lynn_EXCITE_portable')

        # Log the directories being used
        logging.info(f"Source directory: {source_directory}")
        logging.info(f"Destination directory: {destination_directory}")

        # Initialize cohort
        logging.info("Initializing cohort folder")
        cohort = create_cohort_folder(source_directory)

        # Execute the processing and rsync process
        logging.info("Starting file processing and transfer")
        process_and_rsync_files(source_directory, destination_directory, cohort)

        logging.info("Script completed. Check the log file for details: " + str(log_file))
        print(Fore.GREEN + f"Processing complete. Log file: {log_file}" + Style.RESET_ALL)

    except Exception as e:
        logging.error(f"Fatal error in main function:\n{str(e)}\n{traceback.format_exc()}")
        print(Fore.RED + "Fatal error occurred. Check the log file for details." + Style.RESET_ALL)

if __name__ == "__main__":
    main()