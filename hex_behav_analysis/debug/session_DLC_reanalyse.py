from hex_behav_analysis.utils.Session_nwb import Session
from pathlib import Path
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.analysis_manager_arduinoDAQ import Process_Raw_Behaviour_Data
import logging
import os

def logging_setup(cohort_directory):
    # ---- Logging setup -----
    logger = logging.getLogger(__name__)        # Create a logger object
    logger.setLevel(logging.DEBUG)
    log_dir = cohort_directory / 'logs'        # Create a file handler to log messages to a file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir / 'error.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)     # Set this handler to log only errors
    console_handler = logging.StreamHandler()       # Create a console handler to log messages to the console
    console_handler.setLevel(logging.DEBUG)  # This handler will log all levels
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')        # Create a formatter and set it for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)     # Add the handlers to the logger
    logger.addHandler(console_handler)

    return logger



test_cohort = Path(r"/cephfs2/dwelch/Behaviour/test_2")

cohort = Cohort_folder(test_cohort, multi=True, OEAB_legacy=False)

logger = logging_setup(test_cohort)

session = cohort.get_session("250211_110245_wtjp280-4a")

Process_Raw_Behaviour_Data(session, logger=logger)