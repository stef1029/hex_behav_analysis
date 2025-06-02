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
from hex_behav_analysis.utils.analysis_logger import AnalysisLogger




class Process_Raw_Behaviour_Data:
    def __init__(self, session_info, logger, sync_with_ephys=False):
        self.session = session_info
        self.sync_with_ephys = sync_with_ephys  # New parameter
        self.session_id = self.session.get("session_id")
        self.mouse_id = self.session.get("mouse_id")

        self.print = AnalysisLogger()
        self.print.set_verbose(True)  # Set to False to disable verbose logging
        
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
        func_name = "ingest_behaviour_data"
        start_time = time.perf_counter()

        self.data_folder_path = Path(self.session.get("directory"))

        self.raw_video_Path = Path(self.session.get("raw_data")["raw_video"])
        self.behaviour_data_Path = Path(self.session.get("raw_data")["behaviour_data"])
        self.tracker_data_Path = Path(self.session.get("raw_data")["tracker_data"])
        self.arduino_DAQ_Path = Path(self.session.get("raw_data")["arduino_DAQ_h5"])
        # self.OEAB_folder = Path(self.session.get("raw_data")["OEAB"])

        self.print.log(func_name, "Files found, processing...")

        # Look for ephys sync timestamps if enabled
        if self.sync_with_ephys:
            found_timestamps = self.load_ephys_timestamps()
            if found_timestamps:
                self.print.log(func_name, "Ephys timestamps used for synchronization.")

        self.sendkey_logs = Session(self.behaviour_data_Path)
        try:
            self.rig_id = int(self.sendkey_logs.rig_id)
        except:
            self.rig_id = 1
            self.print.log(func_name, "Could not parse rig_id as integer, using default value 1")

        self.video_fps = self.sendkey_logs.video_fps
        
        session_metadata = {"rig_id": self.rig_id, 
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

        # Initialize and run the plotting class
        self.print.log(func_name, "Generating DAQ plots...")
        daq_plotter = DAQ_plot(DAQ_h5_path=self.arduino_DAQ_Path,
                            directory=self.data_folder_path,
                            scales_data=self.scales_data,
                            debug=True)

        self.sendkey_dataframe = self.sendkey_logs.dataframe()

        self.sendkey_logs_filename = self.behaviour_data_Path.parent / f"{self.session_id}_sendkey_logs.csv"
        # with open(self.sendkey_logs_filename, 'w') as f:
        #     json.dump(self.sendkey_dataframe.to_json(orient = "table"), f, indent = 4)
        self.sendkey_dataframe.to_csv(self.sendkey_logs_filename, index=False)
        self.print.log(func_name, f"Saved sendkey logs to {self.sendkey_logs_filename}")

        self.print.log(func_name, "Converting data to NWB format...")
        DAQ_to_nwb(DAQ_h5_path=self.arduino_DAQ_Path, 
                scales_data=self.scales_data,
                session_ID=self.session_id, 
                mouse_id=self.mouse_id, 
                video_directory=self.raw_video_Path, 
                video_timestamps=self.frame_times,  # This now contains either original or ephys timestamps
                session_directory=self.data_folder_path,
                session_metadata=session_metadata,
                session_description="Red Hex behaviour", 
                experimenter="Stefan Rogers-Coltman", 
                institution="MRC LMB", 
                lab="Tripodi Lab",
                max_frame_id=self.max_frame_ID)

        # Calculate time taken
        elapsed_time = time.perf_counter() - start_time
        minutes = round(elapsed_time // 60)
        seconds = round(elapsed_time % 60)
        
        self.print.log(func_name, "Processing complete.")
        self.print.log(func_name, f"Time taken: {minutes} minutes, {seconds} seconds")

    def load_ephys_timestamps(self, pulse_interval=50, target_pin=0):
        """
        Looks for an HDF5 file containing 'ephys_sync_timestamps' in the session folder
        and loads the timestamps with interpolation to fill missing pulses.
        
        Only works with HDF5 (.h5) files.
        
        Args:
            pulse_interval (int): Number of message reads by arduinoDAQ per sync pulse sent.
        
        Returns:
            bool: True if timestamps were successfully loaded, False otherwise.
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
            # Initialize the data structure
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
                                for j in range(1, pulse_interval):  # 1 to pulses_per_gap (not including last which is the next recorded pulse)
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
                self.print.log(func_name, f"Using pin {target_pin} for timestamp synchronization")

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
                # This fixes the "only integer scalar arrays can be converted to a scalar index" error
                self.ephys_timestamps_array = np.array(pin_timestamps)
                
                self.print.log(func_name, f"Created compatible timestamp format with {len(pin_timestamps)} timestamps and sequential message IDs")
                
                return True  # Successfully found and loaded
        except Exception as e:
            self.print.error(func_name, f"Error loading and interpolating ephys sync timestamps: {e}")
            import traceback
            traceback.print_exc()
            
            # Initialize empty data structures to prevent further errors
            self.ephys_timestamps = {"message_ids": [], "timestamps": []}
            self.ephys_timestamps_dict = {}
            self.ephys_timestamps_array = np.array([])  # Empty numpy array
            return False  # Failed to load

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

    def get_camera_frame_times(self):
        func_name = "get_camera_frame_times"
        arduinoDAQh5 = self.arduino_DAQ_Path

        with h5py.File(arduinoDAQh5, 'r') as h5f:
            # Load the 'CAMERA' channel data
            camera_data = np.array(h5f['channel_data']['CAMERA'])
            
            # Decide which timestamps to use
            if self.sync_with_ephys and self.ephys_timestamps is not None:
                # Create a mapping from message_id to ephys timestamp
                message_ids = np.array(h5f['message_ids'])
                ephys_ts_dict = {int(msg_id): ts for msg_id, ts in zip(
                    self.ephys_timestamps["message_ids"], 
                    self.ephys_timestamps["timestamps"]
                )}
                
                # Use ephys timestamps instead of the original ones
                self.timestamps = np.array([ephys_ts_dict.get(int(msg_id), 0) for msg_id in message_ids])
                self.print.log(func_name, "Using ephys synchronized timestamps")
            else:
                # Use the original timestamps
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
            self.print.log(func_name, f"fps set to {fps}")
        else:
            fps = 30
            self.print.log(func_name, "fps set to 30")
        
        # check if frame_IDs is a key:
        if "frame_IDs" in self.video_metadata:
            self.frame_IDs = self.video_metadata["frame_IDs"]
        else:
            error_msg = "Error: No frame_IDs found in tracker_data.json. Processing aborted."
            self.print.error(func_name, error_msg)
            raise Exception(error_msg)
        self.max_frame_ID = max(self.frame_IDs)

        for i, pulse in enumerate(self.camera_pulses):
            self.pulse_times[i] = pulse
        
        if len(self.camera_pulses) < self.max_frame_ID + 1:  # Add +1 because indices start at 0
            new_max_frame_id = len(self.camera_pulses) - 1
            self.frame_IDs = [f for f in self.frame_IDs if f <= new_max_frame_id]
            self.print.log(
                func_name,
                f"Warning: Only {len(self.camera_pulses)} camera pulses recorded, "
                f"so truncating frame IDs to {len(self.frame_IDs)}."
            )
            # Add this debug statement to verify truncation worked
            self.max_frame_ID = max(self.frame_IDs) if self.frame_IDs else -1
            self.print.log(func_name, f"After truncation - New max_frame_ID: {self.max_frame_ID}")
            
        self.print.log(func_name, f"Debug: len pulse times: {len(self.pulse_times)}, len frame IDs: {len(self.frame_IDs)}")
        frame_ID = 0
        for frame_ID in self.frame_IDs:
            self.frame_times[frame_ID] = self.pulse_times[frame_ID]

        # Print video length (using frame rate and frame count), in minutes and seconds, rounded:
        minutes = round(self.true_video_framecount / fps) // 60
        seconds = round(self.true_video_framecount / fps) % 60
        self.print.log(func_name, f"Video length: {minutes} minutes, {seconds} seconds")

        # Calculate percentage of dropped frames:
        dropped_frames = ((len(self.camera_pulses) - self.true_video_framecount) / len(self.camera_pulses)) * 100

        # Print details about dropped frames:
        self.print.log(func_name, f"Length camera pulses: {len(self.camera_pulses)}, length frames: {self.true_video_framecount}, len frame ids: {len(self.frame_IDs)}")

        if dropped_frames >= 40:
            error_msg = f"Error: Too many dropped frames detected ({round(dropped_frames, 1)}%). Processing aborted."
            self.print.error(func_name, error_msg)
            raise Exception(error_msg)

        self.print.log(func_name, f"Percentage dropped frames: {dropped_frames}%")

        self.video_frame_times_filename = self.behaviour_data_Path.parent / f"{self.behaviour_data_Path.name[:13]}_video_frame_times.json"

        output_data = {
            "frame_times": self.frame_times,
            "no_dropped_frames": (self.true_video_framecount - len(self.camera_pulses))
        }

        with open(self.video_frame_times_filename, 'w') as f:
            json.dump(output_data, f, indent=4)

    def get_scales_data(self):
        """
        Retrieves scales data from the sendkey logs and assigns it to the scales_data attribute.
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

        # Initialize the scales_data dictionary with consistent keys
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
            if self.sync_with_ephys and self.ephys_timestamps is not None:
                # Use ephys timestamps instead of original ones
                message_ids = np.array(h5f['message_ids'])
                ephys_ts_dict = {int(msg_id): ts for msg_id, ts in zip(
                    self.ephys_timestamps["message_ids"], 
                    self.ephys_timestamps["timestamps"]
                )}
                self.timestamps = np.array([ephys_ts_dict.get(int(msg_id), 0) for msg_id in message_ids])
                self.print.log(func_name, "Using ephys synchronized timestamps for data")
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
            for i in range(0, length_timestamps, length_timestamps // length_scales):
                try:
                    new_scales_timestamps.append(self.timestamps[i])
                    if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                        new_ephys_scales_timestamps.append(self.ephys_timestamps_array[i])
                except IndexError:
                    new_scales_timestamps.append(self.timestamps[-1])
                    if self.sync_with_ephys and hasattr(self, 'ephys_timestamps_array'):
                        new_ephys_scales_timestamps.append(self.ephys_timestamps_array[-1])

            # If length of timestamps is longer than number of weight readings, cut off last timestamps to make equal length
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
                self.print.log(func_name, f"Generated {len(ephys_scales_pulses)} ephys-synchronized scale pulse timestamps")

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


    def count_files(self, directory):
        file_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_count += 1
        print(f"Number of files in {directory} is {file_count}")

    def find_file(self, directory, tag):
        file_found = False
        for file in directory.glob('*'):
            if not file_found:
                if tag in file.name:
                    file_found = True
                    return file
        if not file_found:
            raise Exception(f"'{tag}' not found in {directory}")

    def find_dir(self, directory):
        """
        if there is only one directory in the directory, return that directory
        """
        dir_found = False
        for file in directory.glob('*'):
            if not dir_found:
                # if dir and does not contain "DLC":    Makes sure it's not detecting the DLC folder that's generated.
                if file.is_dir() and "DLC" not in file.name:
                    dir_found = True
                    return file
        if not dir_found:
            raise Exception(f"No OEAB directory found in {directory}")



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


def main():

    total_start_time = time.perf_counter()

    cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE")

    logger = logging_setup(cohort_directory)

    Cohort = Cohort_folder(cohort_directory, multi = True, plot=False, OEAB_legacy = False)

    directory_info = Cohort.cohort

    sessions_to_process = []
    num_sessions = 0

    refresh = True
    
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            num_sessions += 1
            # session_directory = Path(directory_info["mice"][mouse]["sessions"][session]["directory"])
            if directory_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                if not directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"] == True or refresh == True:
                    date = session[:6]
                    if int(date) >= 241001:
                        sessions_to_process.append(Cohort.get_session(session))     # uses .get_session to make sure that analysis manager has all the paths right.

    print(f"Processing {len(sessions_to_process)} of {num_sessions} sessions...")


    # D:\Data\September_portable\241017_151131\241017_151132_mtao89-1e\241017_151132_mtao89-1e.nwb

    sessions_to_process = ['241017_151132_mtao89-1e']

    # sessions_to_process = ["241017_151132_wtjx249-4b", "241017_151132_mtao89-1e"]
    sessions_to_process = [Cohort.get_session(session) for session in sessions_to_process]

    for session in sessions_to_process:
        print(f"\n\nProcessing {session.get('directory')}...")
        Process_Raw_Behaviour_Data(session, logger)

    # directory_info = Cohort_folder(cohort_directory, multi = True, plot=False, OEAB_legacy = False).cohort

    # print total time taken in minutes and seconds, rounded to whole numbers
        
    print(f"Total time taken: {round((time.perf_counter() - total_start_time) // 60)} minutes, {round((time.perf_counter() - total_start_time) % 60)} seconds")

if __name__ == "__main__":
    main()






