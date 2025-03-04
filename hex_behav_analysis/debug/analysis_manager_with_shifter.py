from pathlib import Path
import os
import matplotlib.pyplot as plt
import json
import time
import cv2 as cv
from datetime import datetime
import multiprocessing as mp
import subprocess
import struct
import h5py
import traceback
import logging

from hex_behav_analysis.Preliminary_analysis_scripts.session_import import Session
from hex_behav_analysis.Preliminary_analysis_scripts.process_ADC_recordings import process_ADC_Recordings
from hex_behav_analysis.Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import
# from Preliminary_analysis_scripts.deeplabcut_setup import DLC_setup
from hex_behav_analysis.utils.DAQ_plot_ArduinoDAQ import DAQ_plot
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.DAQ_to_nwb_ArduinoDAQ import *
from get_video_shift_from_correlation import get_brightness_sensor_alignment

class Process_Raw_Behaviour_Data:
    def __init__(self, session_info, logger):
        self.session = session_info
        self.logger = logger
        self.session_id = self.session.get("session_id")
        self.mouse_id = self.session.get("mouse_id")

        # self.count_files(self.data_folder_path)

        try:
            self.ingest_behaviour_data()
        except Exception as e:
            print(f"Error processing {self.session.get('directory')}: {e}")
            logger.error(f"Error processing {self.session.get('directory')}: {e}")
            traceback.print_exc()

    def ingest_behaviour_data(self):
        """        
        Expected files:

        overlay.avi,
        raw.avi,
        behaviour_data.json,
        Tracker_data.json,
        ArduinoDAQ.json,
        OEAB folder,
        """

        start_time = time.perf_counter()

        self.data_folder_path = Path(self.session.get("directory"))

        self.raw_video_Path = Path(self.session.get("raw_data")["raw_video"])
        self.behaviour_data_Path = Path(self.session.get("raw_data")["behaviour_data"])
        self.tracker_data_Path = Path(self.session.get("raw_data")["tracker_data"])
        self.arduino_DAQ_Path = Path(self.session.get("raw_data")["arduino_DAQ_h5"])
        # self.OEAB_folder = Path(self.session.get("raw_data")["OEAB"])

        print("Files found, processing...")

        self.sendkey_logs = Session(self.behaviour_data_Path)
        try:
            self.rig_id = int(self.sendkey_logs.rig_id)
        except:
            self.rig_id = 1

        self.video_fps = self.sendkey_logs.video_fps
        
        session_metadata = {"rig_id": self.rig_id, 
                            "mouse_weight": self.sendkey_logs.mouse_weight, 
                            "behaviour_phase": self.sendkey_logs.behaviour_phase,
                            "cue_duration": self.sendkey_logs.cue_duration,
                            "wait_duration": self.sendkey_logs.wait_duration,
                            "video_fps": self.video_fps
                            }

        self.get_camera_frame_times()

        self.get_scales_data()

        # Initialize and run the plotting class
        daq_plotter = DAQ_plot(DAQ_h5_path=self.arduino_DAQ_Path,
                            directory=self.data_folder_path,
                            scales_data=self.scales_data,
                            debug=True)

        self.sendkey_dataframe = self.sendkey_logs.dataframe()

        self.sendkey_logs_filename = self.behaviour_data_Path.parent / f"{self.session_id}_sendkey_logs.csv"
        # with open(self.sendkey_logs_filename, 'w') as f:
        #     json.dump(self.sendkey_dataframe.to_json(orient = "table"), f, indent = 4)
        self.sendkey_dataframe.to_csv(self.sendkey_logs_filename, index=False)

        # Check if we need to apply a time shift based on ROI alignment
        self.check_and_apply_ROI_alignment()

        # Delete existing NWB file if it exists
        existing_nwb = self.data_folder_path / (self.data_folder_path.stem + '.nwb')
        if existing_nwb.exists():
            print(f"Removing existing NWB file: {existing_nwb}")
            existing_nwb.unlink()

        # Create a new NWB file with the (potentially) shifted timestamps
        DAQ_to_nwb(DAQ_h5_path=self.arduino_DAQ_Path, 
                   scales_data=self.scales_data,
                   session_ID=self.session_id, 
                   mouse_id=self.mouse_id, 
                   video_directory=self.raw_video_Path, 
                   video_timestamps=self.frame_times,
                   session_directory=self.data_folder_path,
                   session_metadata=session_metadata,
                   session_description="Red Hex behaviour", 
                   experimenter="Stefan Rogers-Coltman", 
                   institution="MRC LMB", 
                   lab="Tripodi Lab",
                   max_frame_id=self.max_frame_ID)

        print("Processing complete.")
        # print time taken in mimutes and seconds:
        print(f"Time taken: {round((time.perf_counter() - start_time) // 60)} minutes, {round((time.perf_counter() - start_time) % 60)} seconds")

    def check_and_apply_ROI_alignment(self):
        """
        Check if ROI alignment is needed and apply if necessary
        """
        # Check if truncated_start_report folder exists
        truncated_start_dir = self.data_folder_path / "truncated_start_report"
        if not truncated_start_dir.exists():
            print("No truncated_start_report folder found, skipping ROI alignment")
            return
        
        # Check if brightness data file exists (indicator of ROI information)
        brightness_files = list(truncated_start_dir.glob(f"{self.session_id}*_brightness_data.csv"))
        if not brightness_files:
            brightness_files = list(truncated_start_dir.glob("*_brightness_data.csv"))
        
        if not brightness_files:
            print("No brightness data files found, skipping ROI alignment")
            return
        
        print("ROI information found, calculating time shift...")
        
        # Import the alignment function from your script
        
        
        # Calculate the shift
        lag = get_brightness_sensor_alignment(
            session_dir=str(self.data_folder_path),
            sensor_name="SENSOR1",  # Default sensor name
            max_lag_seconds=15.0,   # Maximum lag to search
            save_plots=True,        # Save alignment plots
            verbose=True            # Print detailed information
        )
        
        if lag is None:
            print("Failed to determine lag, no shift will be applied")
            return
        
        print(f"Calculated time shift: {lag:.3f} seconds")
        
        # Apply the shift to video timestamps
        for key in list(self.frame_times.keys()):
            if not isinstance(key, str) or key.isdigit():
                # Apply shift only to numeric frame time entries
                self.frame_times[key] += lag
        
        # Save a signal file indicating the shift
        shift_info = {
            "session_id": self.session_id,
            "lag_seconds": lag,
            "applied_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Video timestamps shifted based on ROI alignment"
        }
        
        shift_file = self.data_folder_path / f"{self.session_id}_timestamp_shift.json"
        with open(shift_file, 'w') as f:
            json.dump(shift_info, f, indent=4)
        
        print(f"Applied timestamp shift of {lag:.3f} seconds. Shift info saved to {shift_file}")

    def clean_DAQ_data(self, DAQ_data):
        """
        Cleans None values from DAQ_data if there was an error in the OEAB recording.
        -> Only effective if the number of OEAB pulses was shorter than the message_ids.
        """
        none_indices = set(i for i, ts in enumerate(DAQ_data["timestamps"]) if ts is None)
        print(f"Length of nones: {len(none_indices)}")
        # for key in DAQ_data.keys():
        #     print(f"Key: {key}, Length: {len(DAQ_data[key])}")

        
        if len(none_indices) > 0:
            print(f"OEAB timestamp error, removing {int((len(none_indices) / len(DAQ_data['timestamps'])) * 100)}% of data.")

            # Retain only the elements not in none_indices
            max_len = len(DAQ_data["timestamps"])
            retain_indices = sorted(set(range(max_len)) - none_indices)
            
            for key in DAQ_data.keys():
                if len(DAQ_data[key]) != max_len:
                    # print(f"Warning: Length mismatch for key '{key}'")
                    continue  # Skip lists that do not match the expected length
                DAQ_data[key] = [DAQ_data[key][i] for i in retain_indices]

        return DAQ_data

    def pulse_ID_sync(self, DAQ_data, pulses):
        # Create a dictionary that maps message ID to timestamp
        # This assumes that pulses list has unique values for every message ID
        pulse_dict = {i: timestamp for i, timestamp in enumerate(pulses)}
        print(f"Number of OEAB pulses: {len(pulses)}")
        print(f"Num DAQ binary messages: {len(DAQ_data['message_ids'])}")
        print(f"Highest message ID: {DAQ_data['message_ids'][-1]}")
        
        # Replace message IDs with corresponding timestamps from the pulse_dict
        timestamps = []
        for message_id in DAQ_data["message_ids"]:
            # if message_id > len(pulses):    # test if the message ID is greater than number of pulses, indicating something weird happened
            #     continue
            # else:
            timestamp = pulse_dict.get(message_id, None)
            timestamps.append(timestamp)
        DAQ_data["timestamps"] = timestamps
        # if None is in timestamps, raise an error:
        # if None in DAQ_data["timestamps"]:
        #     raise Exception("DAQ pulses error, could not assign timestamps to DAQ data")
        # if first timestamp = 0  or last timestamp is greater than 
        # 04/24: ISSUE COMING FROM HERE. IT'S NOT FINDING THE RIGHT INDEX IN THE TIMESTAMPS FOR SOME REASON AND SO DEFAULTING TO ADDING THE PULSE ID INSTEAD.
        # Issue arises from errors in the recording of the DAQ pulses, so cannot be fixed by code. Fixed by raising an error and not allowing the file to generate an 'analysed' folder. (SRC 26/6/24)
        
        # print(f"DAQ error percentage: {len(DAQ_data['timestamps']) / len(pulses)}")
        return DAQ_data

    def get_camera_frame_times(self):
        arduinoDAQh5 = self.arduino_DAQ_Path

        with h5py.File(arduinoDAQh5, 'r') as h5f:
            # Load the 'CAMERA' channel data
            camera_data = np.array(h5f['channel_data']['CAMERA'])

            # Load the timestamps
            self.timestamps = np.array(h5f['timestamps'])
        
        # Detect low-to-high transitions (0 -> 1) in camera data
        low_to_high_transitions = np.where((camera_data[:-1] == 0) & (camera_data[1:] == 1))[0]

        # Get the timestamps corresponding to these transitions
        self.camera_pulses = self.timestamps[low_to_high_transitions + 1] # We add 1 to the indices because the transitions occur between two points, and we want the timestamp of the high (1) point.

        self.pulse_times = {}
        self.frame_times = {}

        # import as dictionary the data in tracker_data.json
        with open(self.tracker_data_Path, 'r') as f:
            self.video_metadata = json.load(f)

        cap = cv.VideoCapture(str(self.raw_video_Path))
        self.true_video_framecount = cap.get(cv.CAP_PROP_FRAME_COUNT)
        cap.release()
        if self.video_fps != None:
            fps = self.video_fps
            print(f"fps set to {fps}")

        else:
            fps = 30
            print("fps set to 30")
        

        # check if frame_IDs is a key:
        if "frame_IDs" in self.video_metadata:
            self.frame_IDs = self.video_metadata["frame_IDs"]
        else:
            raise Exception("Error: No frame_IDs found in tracker_data.json. Processing aborted.")
        self.max_frame_ID = max(self.frame_IDs)

        for i, pulse in enumerate(self.camera_pulses):
            self.pulse_times[i] = pulse

        # if len(self.camera_pulses) < self.frame_IDs[-1]:
        #     print(f"Pulses: {len(self.camera_pulses)}, Frame IDs: {self.frame_IDs[-1]}")
        #     raise Exception("Error: Number of camera pulses is less than the number of frame IDs. Check data.")
        
        if len(self.camera_pulses) < self.frame_IDs[-1]:
            new_max_frame_id = len(self.camera_pulses) - 1
            truncated_frame_ids = [f for f in self.frame_IDs if f <= new_max_frame_id]
            self.frame_IDs = truncated_frame_ids
            print(
                f"Warning: Only {len(self.camera_pulses)} camera pulses recorded, "
                f"so truncating frame IDs to {len(self.frame_IDs)}."
            )

        frame_ID = 0
        for frame_ID in self.frame_IDs:
            self.frame_times[frame_ID] = self.pulse_times[frame_ID]

        # print video length (using frame rate of 30 and frame count), in minutes and seconds, rounded:
        print(f"Video length: {round(self.true_video_framecount / fps) // 60} minutes, {round(self.true_video_framecount / fps) % 60} seconds")

        # print percentage dropped frames:
        dropped_frames = ((len(self.camera_pulses) - self.true_video_framecount) / len(self.camera_pulses)) * 100

        # print details about dropped frames:
        print(f"Length camera pulses: {len(self.camera_pulses)}, length frames: {self.true_video_framecount}, len frame ids: {len(self.frame_IDs)}")

        if dropped_frames >= 40:
            raise Exception(f"Error: Too many dropped frames detected ({round(dropped_frames, 1)}%). Processing aborted.")

        print(f"Percentage dropped frames: {dropped_frames}%")

        self.video_frame_times_filename = self.behaviour_data_Path.parent / f"{self.behaviour_data_Path.name[:13]}_video_frame_times.json"

        self.frame_times["no_dropped_frames"] = (self.true_video_framecount - len(self.camera_pulses))

        with open(self.video_frame_times_filename, 'w') as f:
            json.dump(self.frame_times, f, indent = 4)

    def get_scales_data(self):
        """
        Retrieves scales data from the sendkey logs and assigns it to the scales_data attribute.

        Returns:
            None
        """
        scales_logs = self.sendkey_logs.scales_data

        # Load scales channel data from the HDF5 file
        with h5py.File(self.arduino_DAQ_Path, 'r') as h5f:
            scales_channel_data = np.array(h5f['channel_data']['SCALES'])
            scales_timestamps = np.array(h5f['timestamps'])

        # Determine the type of scales based on the logs
        try:
            scales_type = 'wired' if len(scales_logs[0]) == 3 else 'wireless'
        except IndexError:
            raise Exception("Error: No scales data found in sendkey logs. Processing aborted.")

        # Initialize the scales_data dictionary with consistent keys
        self.scales_data = {
            "timestamps": None,
            "weights": None,
            "pulse_IDs": None,
            "sendkey_timestamps": None,
            "mouse_weight_threshold": None,
            "scales_type": scales_type
        }

        if scales_type == 'wireless':
            length_timestamps = len(self.timestamps)
            length_scales = len(scales_logs)

            new_scales_timestamps = []

            for i in range(0, length_timestamps, length_timestamps // length_scales):
                try:
                    new_scales_timestamps.append(self.timestamps[i])
                except IndexError:
                    new_scales_timestamps.append(self.timestamps[-1])

            # If length of timestamps is longer than number of weight readings, cut off last timestamps to make equal length
            if len(new_scales_timestamps) > len(scales_logs):
                new_scales_timestamps = new_scales_timestamps[:len(scales_logs)]

            self.scales_data["timestamps"] = new_scales_timestamps
            self.scales_data["weights"] = [value_pair[1] for value_pair in scales_logs]
            self.scales_data["sendkey_timestamps"] = [value_pair[0] for value_pair in scales_logs]
            self.scales_data["mouse_weight_threshold"] = self.sendkey_logs.mouse_weight

            print("**Warning: Scales data not accurately timestamped.**")

        elif scales_type == 'wired':
            # Detect pulse transitions in the scales channel data
            low_to_high_transitions = np.where((scales_channel_data[:-1] == 0) & (scales_channel_data[1:] == 1))[0]

            # Get the timestamps for each pulse
            scales_pulses = scales_timestamps[low_to_high_transitions + 1]  # Adding 1 for the high (1) point

            # Extract pulse IDs and weights from scales logs
            pulse_IDs = [value_pair[2] for value_pair in scales_logs]
            weights = [value_pair[1] for value_pair in scales_logs]


            print(f"Num pulses: {len(scales_pulses)}")
            # Match pulse IDs to pulse timestamps
            scales_data_dict = {}
            for pulse_ID, weight in zip(pulse_IDs, weights):
                if pulse_ID < len(scales_pulses):
                    scales_data_dict[pulse_ID] = {
                        "timestamp": scales_pulses[pulse_ID],
                        "weight": weight
                    }
                else:
                    print(f"**Warning: Pulse ID {pulse_ID} exceeds recorded scales pulses. Skipping.**")

            # Prepare the scales_data dictionary
            timestamps = []
            weights = []
            pulse_ids = []
            for pulse_ID, data in scales_data_dict.items():
                timestamps.append(data["timestamp"])
                weights.append(data["weight"])
                pulse_ids.append(pulse_ID)

            self.scales_data["timestamps"] = timestamps if timestamps else None
            self.scales_data["weights"] = weights if weights else None
            self.scales_data["pulse_IDs"] = pulse_ids if pulse_ids else None
            self.scales_data["mouse_weight_threshold"] = self.sendkey_logs.mouse_weight

            print(f"Processed {len(pulse_ids)} scales readings with accurate timestamps.")

        # Final output structure for both scales types
        print(f"Scales data processed. Type: {self.scales_data['scales_type']}.")


def main_MP():
    total_start_time = time.perf_counter()

    # cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE")
    cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/November_cohort")

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
    # --------------------------

    # Also set up a specific logger for alignment operations
    alignment_logger = logging.getLogger('alignment')
    alignment_logger.setLevel(logging.INFO)
    alignment_log_file = log_dir / 'alignment.log'
    alignment_file_handler = logging.FileHandler(alignment_log_file)
    alignment_file_handler.setLevel(logging.INFO)
    alignment_file_handler.setFormatter(formatter)
    alignment_logger.addHandler(alignment_file_handler)
    alignment_logger.addHandler(console_handler)

    Cohort = Cohort_folder(cohort_directory, multi=True, plot=False, OEAB_legacy=False)
    directory_info = Cohort.cohort

    # First, let's check which sessions might need ROI-based timestamp alignment
    sessions_to_process = []
    sessions_with_roi = []
    sessions_without_roi = []
    num_sessions = 0
    
    # Option to force reprocessing even if already analyzed
    refresh = False
    
    # Option to only process sessions with ROI data
    only_process_roi_sessions = True

    session_to_process = "250131_150005_wtjp273-3f"
    
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            num_sessions += 1
            session_dir = Path(directory_info["mice"][mouse]["sessions"][session]["directory"])
            
            # Check if the session has ROI data in the truncated_start_report folder
            truncated_dir = session_dir / "truncated_start_report"
            has_roi_data = (truncated_dir.exists() and 
                           any(f.name.endswith("_brightness_data.csv") 
                               for f in truncated_dir.glob("*")))
            
            # Check if the session needs processing
            needs_processing = (not directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"] or refresh)
            raw_data_present = directory_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"]
            
            # Filter by date if needed (only process sessions after a certain date)
            date = session[:6]
            meets_date_criteria = int(date) >= 241001  # Only process sessions after 2024-10-01
            
            if raw_data_present and needs_processing and meets_date_criteria:
                if has_roi_data:
                    sessions_with_roi.append(session)
                    alignment_logger.info(f"Session {session} has ROI data available for alignment")
                    if not only_process_roi_sessions:
                        sessions_to_process.append(Cohort.get_session(session))
                else:
                    sessions_without_roi.append(session)
                    if not only_process_roi_sessions:
                        sessions_to_process.append(Cohort.get_session(session))
            
            # If we only want to process ROI sessions and this session has ROI data
            if only_process_roi_sessions and raw_data_present and has_roi_data:
                sessions_to_process.append(Cohort.get_session(session))

    # Log summary of what we found
    alignment_logger.info(f"Found {len(sessions_with_roi)} sessions with ROI data")
    alignment_logger.info(f"Found {len(sessions_without_roi)} sessions without ROI data")
    
    # Process the selected sessions
    print(f"Processing {len(sessions_to_process)} of {num_sessions} sessions...")
    alignment_logger.info(f"Processing {len(sessions_to_process)} of {num_sessions} sessions...")
    
    for session in sessions_to_process:
        session_id = session.get('session_id')
        print(f"\n\nProcessing {session.get('directory')}...")

        if session_to_process and session_id != session_to_process:
            print(f"Skipping session {session_id}")
            continue
        
        # Check if this session has ROI data before processing
        session_dir = Path(session.get('directory'))
        truncated_dir = session_dir / "truncated_start_report"
        has_roi = (truncated_dir.exists() and 
                  any(f.name.endswith("_brightness_data.csv") 
                      for f in truncated_dir.glob("*")))
        
        if has_roi:
            alignment_logger.info(f"Starting processing of session {session_id} WITH ROI alignment")
        else:
            alignment_logger.info(f"Starting processing of session {session_id} WITHOUT ROI alignment")
        
        try:
            Process_Raw_Behaviour_Data(session, logger)
            alignment_logger.info(f"Successfully processed session {session_id}")
        except Exception as e:
            alignment_logger.error(f"Failed to process session {session_id}: {str(e)}")
            traceback.print_exc()

    # Refresh directory info to see what was processed
    directory_info = Cohort_folder(cohort_directory, multi=False, plot=False, OEAB_legacy=False).cohort
    
    # Calculate how many sessions were successfully processed
    processed_count = 0
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            if directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"]:
                processed_count += 1
    
    alignment_logger.info(f"Processing complete. Successfully processed {processed_count} of {num_sessions} sessions.")
    
    # print total time taken in minutes and seconds, rounded to whole numbers  
    total_time = time.perf_counter() - total_start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Total time taken: {minutes} minutes, {seconds} seconds")
    alignment_logger.info(f"Total time taken: {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    main_MP()