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
import numpy as np

from hex_behav_analysis.Preliminary_analysis_scripts.session_import import Session
from hex_behav_analysis.Preliminary_analysis_scripts.process_ADC_recordings import process_ADC_Recordings
from hex_behav_analysis.Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import
# from Preliminary_analysis_scripts.deeplabcut_setup import DLC_setup
from hex_behav_analysis.utils.DAQ_plot_ArduinoDAQ import DAQ_plot
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.DAQ_to_nwb_ArduinoDAQ import *
from hex_behav_analysis.utils.analysis_logger import AnalysisLogger
from get_video_shift_from_correlation import get_brightness_sensor_alignment


class Process_Raw_Behaviour_Data:
    def __init__(self, session_info, logger, sync_with_ephys=False):
        """
        Initialise the processing class for raw behaviour data.
        
        Args:
            session_info: Dictionary containing session information
            logger: Logger instance for error logging
            sync_with_ephys: Boolean flag to enable ephys synchronisation
        """
        self.session = session_info
        self.logger = logger
        self.sync_with_ephys = sync_with_ephys
        self.session_id = self.session.get("session_id")
        self.mouse_id = self.session.get("mouse_id")

        self.print = AnalysisLogger()
        self.print.set_verbose(True)  # Set to False to disable verbose logging
        
        try:
            self.ingest_behaviour_data()
        except Exception as e:
            self.print.error("__init__", f"Error processing {self.session.get('directory')}: {e}")
            logger.error(f"Error processing {self.session.get('directory')}: {e}")
            traceback.print_exc()

    def ingest_behaviour_data(self):
        """        
        Process raw behaviour data files and convert to standardised formats.
        
        Expected files:
        - overlay.avi
        - raw.avi
        - behaviour_data.json
        - Tracker_data.json
        - ArduinoDAQ.h5
        - OEAB folder (if applicable)
        """
        func_name = "ingest_behaviour_data"
        start_time = time.perf_counter()

        self.data_folder_path = Path(self.session.get("directory"))

        self.raw_video_Path = Path(self.session.get("raw_data")["raw_video"])
        self.behaviour_data_Path = Path(self.session.get("raw_data")["behaviour_data"])
        self.tracker_data_Path = Path(self.session.get("raw_data")["tracker_data"])
        self.arduino_DAQ_Path = Path(self.session.get("raw_data")["arduino_DAQ_h5"])

        self.print.log(func_name, "Files found, processing...")

        # Look for ephys sync timestamps if enabled
        if self.sync_with_ephys:
            found_timestamps = self.load_ephys_timestamps()
            if found_timestamps:
                self.print.log(func_name, "Ephys timestamps used for synchronisation.")

        self.sendkey_logs = Session(self.behaviour_data_Path)
        try:
            self.rig_id = int(self.sendkey_logs.rig_id)
        except:
            self.rig_id = 1
            self.print.log(func_name, "Could not parse rig_id as integer, using default value 1")

        self.video_fps = self.sendkey_logs.video_fps
        
        session_metadata = {
            "rig_id": self.rig_id, 
            "mouse_weight": self.sendkey_logs.mouse_weight, 
            "behaviour_phase": self.sendkey_logs.behaviour_phase,
            "cue_duration": self.sendkey_logs.cue_duration,
            "wait_duration": self.sendkey_logs.wait_duration,
            "video_fps": self.video_fps
        }

        self.print.log(func_name, "Processing camera frame times...")
        self.get_camera_frame_times()

        self.print.log(func_name, "Processing scales data...")
        self.get_scales_data()

        # Initialise and run the plotting class
        self.print.log(func_name, "Generating DAQ plots...")
        daq_plotter = DAQ_plot(
            DAQ_h5_path=self.arduino_DAQ_Path,
            directory=self.data_folder_path,
            scales_data=self.scales_data,
            debug=True
        )

        self.sendkey_dataframe = self.sendkey_logs.dataframe()

        self.sendkey_logs_filename = self.behaviour_data_Path.parent / f"{self.session_id}_sendkey_logs.csv"
        self.sendkey_dataframe.to_csv(self.sendkey_logs_filename, index=False)
        self.print.log(func_name, f"Saved sendkey logs to {self.sendkey_logs_filename}")

        # Check if we need to apply a time shift based on ROI alignment
        self.check_and_apply_ROI_alignment()

        # Delete existing NWB file if it exists
        existing_nwb = self.data_folder_path / (self.data_folder_path.stem + '.nwb')
        if existing_nwb.exists():
            self.print.log(func_name, f"Removing existing NWB file: {existing_nwb}")
            existing_nwb.unlink()

        # Create a new NWB file with the (potentially) shifted timestamps
        self.print.log(func_name, "Converting data to NWB format...")
        DAQ_to_nwb(
            DAQ_h5_path=self.arduino_DAQ_Path, 
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
            max_frame_id=self.max_frame_ID
        )

        # Calculate time taken
        elapsed_time = time.perf_counter() - start_time
        minutes = round(elapsed_time // 60)
        seconds = round(elapsed_time % 60)
        
        self.print.log(func_name, "Processing complete.")
        self.print.log(func_name, f"Time taken: {minutes} minutes, {seconds} seconds")

    def check_and_apply_ROI_alignment(self):
        """
        Check if ROI alignment is needed and apply timestamp shift if necessary.
        This method looks for brightness data from ROI analysis and calculates
        the temporal shift needed to align video timestamps with sensor data.
        """
        func_name = "check_and_apply_ROI_alignment"
        
        # Check if truncated_start_report folder exists
        truncated_start_dir = self.data_folder_path / "truncated_start_report"
        if not truncated_start_dir.exists():
            self.print.log(func_name, "No truncated_start_report folder found, skipping ROI alignment")
            return
        
        # Check if brightness data file exists (indicator of ROI information)
        brightness_files = list(truncated_start_dir.glob(f"{self.session_id}*_brightness_data.csv"))
        if not brightness_files:
            brightness_files = list(truncated_start_dir.glob("*_brightness_data.csv"))
        
        if not brightness_files:
            self.print.log(func_name, "No brightness data files found, skipping ROI alignment")
            return
        
        self.print.log(func_name, "ROI information found, calculating time shift...")
        
        # Calculate the shift
        lag = get_brightness_sensor_alignment(
            session_dir=str(self.data_folder_path),
            sensor_name="SENSOR1",  # Default sensor name
            max_lag_seconds=15.0,   # Maximum lag to search
            save_plots=True,        # Save alignment plots
            verbose=False           # Print detailed information
        )
        
        if lag is None:
            self.print.log(func_name, "Failed to determine lag, no shift will be applied")
            return
        
        self.print.log(func_name, f"Calculated time shift: {lag:.3f} seconds")
        
        # Apply the shift to video timestamps
        shifted_frames = 0
        for key in list(self.frame_times.keys()):
            if not isinstance(key, str) and isinstance(key, int):
                # Apply shift only to numeric frame time entries
                self.frame_times[key] += lag
                shifted_frames += 1
        
        self.print.log(func_name, f"Applied shift to {shifted_frames} frame timestamps")
        
        # Save a signal file indicating the shift
        shift_info = {
            "session_id": self.session_id,
            "lag_seconds": lag,
            "applied_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Video timestamps shifted based on ROI alignment",
            "shifted_frames": shifted_frames
        }
        
        shift_file = self.data_folder_path / f"{self.session_id}_timestamp_shift.json"
        with open(shift_file, 'w') as f:
            json.dump(shift_info, f, indent=4)
        
        self.print.log(func_name, f"Applied timestamp shift of {lag:.3f} seconds. Shift info saved to {shift_file}")

    def load_ephys_timestamps(self, pulse_interval=50, target_pin=0):
        """
        Load ephys synchronisation timestamps from HDF5 file and interpolate missing pulses.
        
        Args:
            pulse_interval: Number of message reads by arduinoDAQ per sync pulse sent
            target_pin: Pin number to use for timestamp synchronisation
            
        Returns:
            bool: True if timestamps were successfully loaded, False otherwise
        """
        func_name = "load_ephys_timestamps"
        
        # Look specifically for HDF5 files with ephys_sync_timestamps in the name
        timestamp_files = list(self.data_folder_path.glob("*ephys_sync_timestamps*.h5"))
        
        if not timestamp_files:
            self.print.log(func_name, "No ephys sync timestamp HDF5 file found. Proceeding with default timestamps.")
            return False
        
        timestamp_file = timestamp_files[0]  # Take the first matching file
        self.print.log(func_name, f"Found ephys sync timestamp file: {timestamp_file}")
        
        try:
            # Initialise the data structure
            timestamp_data = {"pins": {}}
            
            # Load HDF5 file
            with h5py.File(timestamp_file, 'r') as f:
                # Check if the file has a 'pins' group
                if 'pins' in f:
                    pins_group = f['pins']
                    for pin_name in pins_group:
                        pin_number = int(pin_name.replace('pin_', ''))
                        pin_group = pins_group[pin_name]
                        
                        # Read high_timestamps and durations
                        if 'high_timestamps' in pin_group and 'durations' in pin_group:
                            high_timestamps = np.array(pin_group['high_timestamps'])
                            durations = np.array(pin_group['durations'])
                            
                            # Convert to the expected format
                            events = []
                            for ts, dur in zip(high_timestamps, durations):
                                events.append({"high_timestamp": float(ts), "duration": float(dur)})
                            
                            timestamp_data["pins"][str(pin_number)] = events
                
                # Extract metadata if present
                if 'metadata' in f:
                    metadata = {}
                    for key, value in f['metadata'].attrs.items():
                        metadata[key] = value
                    timestamp_data["metadata"] = metadata
            
            # Process the data which now has high_timestamp and duration for each pin
            self.ephys_timestamps = {}  # This will hold pin-specific data
            
            # Extract metadata if present
            if "metadata" in timestamp_data:
                self.ephys_metadata = timestamp_data["metadata"]
                    
            # Extract pin data and interpolate missing pulses
            recorded_pulses_counts = {}
            all_timestamps = []  # Will collect all timestamps for all pins
            if "pins" in timestamp_data:
                # First, process pins data and collect timestamps
                for pin, events in timestamp_data["pins"].items():
                    # Convert pin string to integer
                    pin_int = int(pin)
                    
                    # Get the high timestamps and durations from the loaded data
                    recorded_pulses = []
                    for event in events:
                        # Format has high_timestamp and duration
                        if "high_timestamp" in event and "duration" in event:
                            high_time = event["high_timestamp"]
                            duration = event["duration"]
                            recorded_pulses.append((high_time, duration))
                    
                    recorded_pulses_counts[pin_int] = len(recorded_pulses)
                    self.print.log(func_name, f"Original pin {pin_int} pulse count: {len(recorded_pulses)}")
                    
                    # Sort pulses by timestamp to ensure they're in chronological order
                    recorded_pulses.sort(key=lambda x: x[0])
                    
                    # If we have at least 2 pulses, we can interpolate between them
                    if len(recorded_pulses) >= 2:
                        interpolated_pulses = []
                        
                        # For each pair of consecutive recorded pulses
                        for i in range(len(recorded_pulses) - 1):
                            current_pulse = recorded_pulses[i]
                            next_pulse = recorded_pulses[i + 1]
                            
                            # Add the current pulse to our interpolated list
                            interpolated_pulses.append(current_pulse)
                            
                            # Calculate time between pulses
                            time_gap = next_pulse[0] - current_pulse[0]
                            
                            # Calculate the average duration for interpolated pulses
                            avg_duration = (current_pulse[1] + next_pulse[1]) / 2
                            
                            # If time gap is large enough to interpolate pulses
                            if time_gap > 0:                                
                                # Time interval between interpolated pulses
                                interval = time_gap / pulse_interval
                                
                                # Generate interpolated pulses
                                for j in range(1, pulse_interval):
                                    interp_time = current_pulse[0] + (interval * j)
                                    interpolated_pulses.append((interp_time, avg_duration))
                        
                        # Add the last recorded pulse
                        if recorded_pulses:
                            interpolated_pulses.append(recorded_pulses[-1])
                        
                        # Store the interpolated pulses
                        self.ephys_timestamps[pin_int] = interpolated_pulses
                        
                        # Collect all timestamps for this pin
                        all_timestamps.extend([ts for ts, _ in interpolated_pulses])
                    else:
                        # Not enough pulses to interpolate, use as-is
                        self.ephys_timestamps[pin_int] = recorded_pulses
                        all_timestamps.extend([ts for ts, _ in recorded_pulses])
                
                # Print statistics
                pin_stats = {pin: len(events) for pin, events in self.ephys_timestamps.items()}
                self.print.log(func_name, "Successfully loaded and interpolated ephys sync timestamps:")
                for pin, count in pin_stats.items():
                    original_count = recorded_pulses_counts.get(pin, 0)
                    self.print.log(func_name, f"  Pin {pin}: {count} total pulses ({original_count} recorded, {count - original_count} interpolated)")
                
                # Get DAQ message count for message ID generation
                daq_message_count = 0
                try:
                    with h5py.File(self.arduino_DAQ_Path, 'r') as h5f:
                        daq_message_ids = np.array(h5f['message_ids'])
                        daq_message_count = len(daq_message_ids)
                        self.print.log(func_name, f"DAQ message count: {daq_message_count}")
                        
                        # Compare with ephys pulse counts
                        for pin, count in recorded_pulses_counts.items():
                            if daq_message_count > 0:
                                ratio = daq_message_count / count
                                self.print.log(func_name, f"Comparison - Pin {pin}: {count * pulse_interval} recorded pulses vs {daq_message_count} DAQ messages")
                                self.print.log(func_name, f"  Ratio (messages/pulses): {ratio:.6f} ({ratio*100:.2f}%)")

                except Exception as e:
                    self.print.log(func_name, f"Could not read DAQ message count: {e}")
                
                # Now create a dictionary format that's compatible with get_camera_frame_times and other functions
                self.print.log(func_name, f"Using pin {target_pin} for timestamp synchronisation")

                if target_pin not in self.ephys_timestamps or len(self.ephys_timestamps[target_pin]) == 0:
                    error_msg = f"No pulses found on target pin {target_pin}. Session processing stopped."
                    self.print.error(func_name, error_msg)
                    raise ValueError(error_msg)
                
                # Extract timestamps from the chosen pin
                pin_timestamps = [ts for ts, _ in self.ephys_timestamps[target_pin]]
                
                # Generate sequential message IDs from 0 to n-1
                message_ids = list(range(len(pin_timestamps)))
                
                # Create the dictionary format expected by other functions
                self.ephys_timestamps_dict = {}
                for i, ts in enumerate(pin_timestamps):
                    self.ephys_timestamps_dict[i] = ts
                
                # Adding these keys that are expected by get_camera_frame_times
                self.ephys_timestamps["message_ids"] = message_ids
                self.ephys_timestamps["timestamps"] = pin_timestamps
                
                # Create timestamp array as NUMPY ARRAY for compatibility with array indexing
                self.ephys_timestamps_array = np.array(pin_timestamps)
                
                self.print.log(func_name, f"Created compatible timestamp format with {len(pin_timestamps)} timestamps and sequential message IDs")
                
                return True  # Successfully found and loaded
        except Exception as e:
            self.print.error(func_name, f"Error loading and interpolating ephys sync timestamps: {e}")
            import traceback
            traceback.print_exc()
            
            # Initialise empty data structures to prevent further errors
            self.ephys_timestamps = {"message_ids": [], "timestamps": []}
            self.ephys_timestamps_dict = {}
            self.ephys_timestamps_array = np.array([])  # Empty numpy array
            return False  # Failed to load

    def get_camera_frame_times(self):
        """
        Extract camera frame timestamps from Arduino DAQ data and align with video frames.
        Handles dropped frames and creates a mapping between frame IDs and timestamps.
        """
        func_name = "get_camera_frame_times"
        arduinoDAQh5 = self.arduino_DAQ_Path

        with h5py.File(arduinoDAQh5, 'r') as h5f:
            # Load the 'CAMERA' channel data
            camera_data = np.array(h5f['channel_data']['CAMERA'])
            
            # Decide which timestamps to use
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps') and self.ephys_timestamps is not None:
                # Create a mapping from message_id to ephys timestamp
                message_ids = np.array(h5f['message_ids'])
                ephys_ts_dict = {int(msg_id): ts for msg_id, ts in zip(
                    self.ephys_timestamps["message_ids"], 
                    self.ephys_timestamps["timestamps"]
                )}
                
                # Use ephys timestamps instead of the original ones
                self.timestamps = np.array([ephys_ts_dict.get(int(msg_id), 0) for msg_id in message_ids])
                self.print.log(func_name, "Using ephys synchronised timestamps")
            else:
                # Use the original timestamps
                self.timestamps = np.array(h5f['timestamps'])
        
        # Detect low-to-high transitions (0 -> 1) in camera data
        low_to_high_transitions = np.where((camera_data[:-1] == 0) & (camera_data[1:] == 1))[0]

        # Get the timestamps corresponding to these transitions
        self.camera_pulses = self.timestamps[low_to_high_transitions + 1]  # Add 1 for the high (1) point

        self.pulse_times = {}
        self.frame_times = {}

        # Import video metadata
        with open(self.tracker_data_Path, 'r') as f:
            self.video_metadata = json.load(f)

        cap = cv.VideoCapture(str(self.raw_video_Path))
        self.true_video_framecount = cap.get(cv.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if self.video_fps is not None:
            fps = self.video_fps
            self.print.log(func_name, f"fps set to {fps}")
        else:
            fps = 30
            self.print.log(func_name, "fps set to 30")
        
        # Check if frame_IDs is a key
        if "frame_IDs" in self.video_metadata:
            self.frame_IDs = self.video_metadata["frame_IDs"]
        else:
            error_msg = "Error: No frame_IDs found in tracker_data.json. Processing aborted."
            self.print.error(func_name, error_msg)
            raise Exception(error_msg)
        
        self.max_frame_ID = max(self.frame_IDs)

        # Create pulse_times dictionary
        for i, pulse in enumerate(self.camera_pulses):
            self.pulse_times[i] = pulse
        
        # CRITICAL FIX: Ensure frame_IDs don't exceed available camera pulses
        original_frame_count = len(self.frame_IDs)
        available_pulses = len(self.camera_pulses)
        
        if self.max_frame_ID >= available_pulses:
            # Truncate frame_IDs to fit available pulses
            self.frame_IDs = [f for f in self.frame_IDs if f < available_pulses]
            self.max_frame_ID = max(self.frame_IDs) if self.frame_IDs else -1
            
            self.print.log(
                func_name,
                f"Truncated frame_IDs from {original_frame_count} to {len(self.frame_IDs)} "
                f"to match available camera pulses ({available_pulses})"
            )
            
        # Ensure we don't exceed available pulses when creating frame_times
        valid_frame_count = 0
        for frame_ID in self.frame_IDs:
            if frame_ID < len(self.pulse_times):
                self.frame_times[frame_ID] = self.pulse_times[frame_ID]
                valid_frame_count += 1
            else:
                self.print.log(func_name, f"Skipping frame_ID {frame_ID} - no corresponding pulse")
        
        self.print.log(func_name, f"Created {valid_frame_count} valid frame timestamps")
        self.print.log(func_name, f"Max frame_ID after processing: {self.max_frame_ID}")
        
        # Update max_frame_ID to reflect actual data
        if self.frame_times:
            self.max_frame_ID = max(self.frame_times.keys())
        else:
            self.max_frame_ID = -1
            
        self.print.log(func_name, f"Final max_frame_ID: {self.max_frame_ID}")

        # Print video length
        minutes = round(self.true_video_framecount / fps) // 60
        seconds = round(self.true_video_framecount / fps) % 60
        self.print.log(func_name, f"Video length: {minutes} minutes, {seconds} seconds")

        # Calculate percentage of dropped frames
        if len(self.camera_pulses) > 0:
            dropped_frames = ((len(self.camera_pulses) - self.true_video_framecount) / len(self.camera_pulses)) * 100
        else:
            dropped_frames = 100
            
        self.print.log(func_name, f"Camera pulses: {len(self.camera_pulses)}, Video frames: {self.true_video_framecount}, Valid frame_IDs: {len(self.frame_IDs)}")

        if dropped_frames >= 40:
            error_msg = f"Error: Too many dropped frames detected ({round(dropped_frames, 1)}%). Processing aborted."
            self.print.error(func_name, error_msg)
            raise Exception(error_msg)

        self.print.log(func_name, f"Percentage dropped frames: {dropped_frames:.1f}%")

        # Save frame times
        self.video_frame_times_filename = self.behaviour_data_Path.parent / f"{self.behaviour_data_Path.name[:13]}_video_frame_times.json"

        output_data = {
            "frame_times": self.frame_times,
            "no_dropped_frames": int(self.true_video_framecount - len(self.camera_pulses)),
            "max_frame_id": self.max_frame_ID,
            "total_valid_frames": len(self.frame_times)
        }

        with open(self.video_frame_times_filename, 'w') as f:
            json.dump(output_data, f, indent=4)

    def get_scales_data(self):
        """
        Retrieve scales data from sendkey logs and align with timestamps.
        Handles both wireless and wired scales types.
        """
        func_name = "get_scales_data"
        scales_logs = self.sendkey_logs.scales_data

        # Determine the type of scales based on the logs before trying to access the HDF5 file
        try:
            scales_type = 'wired' if len(scales_logs[0]) == 3 else 'wireless'
        except IndexError:
            error_msg = "Error: No scales data found in sendkey logs. Processing aborted."
            self.print.error(func_name, error_msg)
            raise Exception(error_msg)
            
        self.print.log(func_name, f"Detected scales type: {scales_type}")

        # Initialise the scales_data dictionary with consistent keys
        self.scales_data = {
            "timestamps": None,
            "weights": None,
            "pulse_IDs": None,
            "sendkey_timestamps": None,
            "mouse_weight_threshold": None,
            "scales_type": scales_type
        }
        
        # Add ephys key if available
        if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
            self.scales_data["ephys_timestamps"] = None

        # Get timestamps from the HDF5 file - needed for both types
        with h5py.File(self.arduino_DAQ_Path, 'r') as h5f:
            # Decide which timestamps to use
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps') and self.ephys_timestamps is not None:
                # Use ephys timestamps instead of original ones
                message_ids = np.array(h5f['message_ids'])
                ephys_ts_dict = {int(msg_id): ts for msg_id, ts in zip(
                    self.ephys_timestamps["message_ids"], 
                    self.ephys_timestamps["timestamps"]
                )}
                self.timestamps = np.array([ephys_ts_dict.get(int(msg_id), 0) for msg_id in message_ids])
                self.print.log(func_name, "Using ephys synchronised timestamps for scales data")
            else:
                # Use original timestamps
                self.timestamps = np.array(h5f['timestamps'])
            
            # Only try to get SCALES channel data if scales_type is wired
            if scales_type == 'wired':
                try:
                    # Check if SCALES channel exists first
                    if 'SCALES' in h5f['channel_data']:
                        scales_channel_data = np.array(h5f['channel_data']['SCALES'])
                    else:
                        self.print.warning(func_name, "SCALES channel not found in HDF5 file. Treating as wireless scales.")
                        scales_type = 'wireless'  # Override to wireless if SCALES channel doesn't exist
                        self.scales_data["scales_type"] = 'wireless'
                except Exception as e:
                    self.print.warning(func_name, f"Error accessing SCALES channel: {e}. Treating as wireless scales.")
                    scales_type = 'wireless'  # Override to wireless if there's an error
                    self.scales_data["scales_type"] = 'wireless'

        if scales_type == 'wireless':
            length_timestamps = len(self.timestamps)
            length_scales = len(scales_logs)

            new_scales_timestamps = []
            new_ephys_scales_timestamps = [] if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array') else None

            # Generate evenly spaced timestamps based on the available timestamps
            for i in range(0, length_timestamps, length_timestamps // length_scales if length_scales > 0 else 1):
                try:
                    new_scales_timestamps.append(self.timestamps[i])
                    if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                        new_ephys_scales_timestamps.append(self.ephys_timestamps_array[i])
                except IndexError:
                    new_scales_timestamps.append(self.timestamps[-1])
                    if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                        new_ephys_scales_timestamps.append(self.ephys_timestamps_array[-1])

            # If length of timestamps is longer than number of weight readings, cut off last timestamps
            if len(new_scales_timestamps) > len(scales_logs):
                new_scales_timestamps = new_scales_timestamps[:len(scales_logs)]
                if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                    new_ephys_scales_timestamps = new_ephys_scales_timestamps[:len(scales_logs)]

            self.scales_data["timestamps"] = new_scales_timestamps
            self.scales_data["weights"] = [value_pair[1] for value_pair in scales_logs]
            self.scales_data["sendkey_timestamps"] = [value_pair[0] for value_pair in scales_logs]
            self.scales_data["mouse_weight_threshold"] = self.sendkey_logs.mouse_weight
            
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                self.scales_data["ephys_timestamps"] = new_ephys_scales_timestamps
                self.print.log(func_name, "Added ephys timestamps to wireless scales data")

            self.print.log(func_name, "**Warning: Wireless scales data not accurately timestamped.**")

        elif scales_type == 'wired':
            # We already verified SCALES channel exists for wired type
            with h5py.File(self.arduino_DAQ_Path, 'r') as h5f:
                scales_channel_data = np.array(h5f['channel_data']['SCALES'])
                scales_timestamps = self.timestamps  # Use the timestamps we got earlier
            
            # Detect pulse transitions in the scales channel data
            low_to_high_transitions = np.where((scales_channel_data[:-1] == 0) & (scales_channel_data[1:] == 1))[0]

            # Filter transitions to ensure they're within bounds of the ephys timestamps array
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                max_index = len(self.ephys_timestamps_array) - 1
                low_to_high_transitions = low_to_high_transitions[low_to_high_transitions + 1 <= max_index]
                
            # Get the timestamps for each pulse
            scales_pulses = scales_timestamps[low_to_high_transitions + 1]  # Adding 1 for the high (1) point
            
            # If ephys timestamps available, get those too
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                ephys_scales_pulses = self.ephys_timestamps_array[low_to_high_transitions + 1]
                self.print.log(func_name, f"Generated {len(ephys_scales_pulses)} ephys-synchronised scale pulse timestamps")

            # Extract pulse IDs and weights from scales logs
            pulse_IDs = [value_pair[2] for value_pair in scales_logs]
            weights = [value_pair[1] for value_pair in scales_logs]

            self.print.log(func_name, f"Num pulses: {len(scales_pulses)}")
            # Match pulse IDs to pulse timestamps
            scales_data_dict = {}
            skipped_pulse_count = 0  # Counter for skipped pulses
            
            for pulse_ID, weight in zip(pulse_IDs, weights):
                if pulse_ID < len(scales_pulses):
                    scales_data_dict[pulse_ID] = {
                        "timestamp": scales_pulses[pulse_ID],
                        "weight": weight
                    }
                    # Add ephys timestamp if available
                    if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                        if pulse_ID < len(ephys_scales_pulses):
                            scales_data_dict[pulse_ID]["ephys_timestamp"] = ephys_scales_pulses[pulse_ID]
                else:
                    skipped_pulse_count += 1  # Increment counter instead of printing
            
            # Print a single summary line for skipped pulses
            if skipped_pulse_count > 0:
                self.print.log(func_name, f"**Warning: {skipped_pulse_count} pulse IDs exceed recorded scales pulses and were skipped.**")

            # Prepare the scales_data dictionary
            timestamps = []
            weights = []
            pulse_ids = []
            ephys_timestamps = [] if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array') else None
            
            for pulse_ID, data in scales_data_dict.items():
                timestamps.append(data["timestamp"])
                weights.append(data["weight"])
                pulse_ids.append(pulse_ID)
                if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array') and "ephys_timestamp" in data:
                    ephys_timestamps.append(data["ephys_timestamp"])

            self.scales_data["timestamps"] = timestamps if timestamps else None
            self.scales_data["weights"] = weights if weights else None
            self.scales_data["pulse_IDs"] = pulse_ids if pulse_ids else None
            self.scales_data["mouse_weight_threshold"] = self.sendkey_logs.mouse_weight
            
            if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                self.scales_data["ephys_timestamps"] = ephys_timestamps if ephys_timestamps else None
                self.print.log(func_name, f"Added {len(ephys_timestamps) if ephys_timestamps else 0} ephys timestamps to wired scales data")

            self.print.log(func_name, f"Processed {len(pulse_ids)} scales readings with accurate timestamps.")

        # Final output structure for both scales types
        self.print.log(func_name, f"Scales data processed. Type: {self.scales_data['scales_type']}.")


def logging_setup(cohort_directory):
    """
    Set up logging configuration for the analysis.
    
    Args:
        cohort_directory: Path to the cohort directory
        
    Returns:
        logger: Configured logger instance
    """
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory
    log_dir = cohort_directory / 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create file handler for errors
    log_file = log_dir / 'error.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main_MP():
    """
    Main function for multiprocessing analysis of behaviour data.
    Supports ROI-based timestamp alignment when available.
    """
    cohorts = []
    cohort_directory = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/DATA_ArduinoDAQ")
    cohorts.append(cohort_directory)
    
    for cohort_directory in cohorts:
        total_start_time = time.perf_counter()
        
        # Set up logging
        logger = logging_setup(cohort_directory)
        
        # Also set up a specific logger for alignment operations
        alignment_logger = logging.getLogger('alignment')
        alignment_logger.setLevel(logging.INFO)
        alignment_log_file = cohort_directory / 'logs' / 'alignment.log'
        alignment_file_handler = logging.FileHandler(alignment_log_file)
        alignment_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        alignment_file_handler.setFormatter(formatter)
        alignment_logger.addHandler(alignment_file_handler)
        alignment_logger.addHandler(logging.StreamHandler())

        Cohort = Cohort_folder(cohort_directory, multi=True, plot=False, OEAB_legacy=False)
        directory_info = Cohort.cohort

        # First, check which sessions might need ROI-based timestamp alignment
        sessions_to_process = []
        sessions_with_roi = []
        sessions_without_roi = []
        num_sessions = 0
        
        # Option to force reprocessing even if already analysed
        refresh = True
        
        # Option to only process sessions with ROI data
        only_process_roi_sessions = True
        
        # Option to enable ephys synchronisation
        sync_with_ephys = False

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
                        if not only_process_roi_sessions or only_process_roi_sessions:
                            sessions_to_process.append(Cohort.get_session(session))
                    else:
                        sessions_without_roi.append(session)
                        if not only_process_roi_sessions:
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
                Process_Raw_Behaviour_Data(session, logger, sync_with_ephys=sync_with_ephys)
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
        
        # Print total time taken
        total_time = time.perf_counter() - total_start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"Total time taken: {minutes} minutes, {seconds} seconds")
        alignment_logger.info(f"Total time taken: {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main_MP()