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

from Preliminary_analysis_scripts.session_import import Session
from Preliminary_analysis_scripts.process_ADC_recordings import process_ADC_Recordings
from Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import
# from Preliminary_analysis_scripts.deeplabcut_setup import DLC_setup
from DAQ_plot_ArduinoDAQ import DAQ_plot
from Cohort_folder import Cohort_folder
from DAQ_to_nwb_ArduinoDAQ import *




class Process_Raw_Behaviour_Data:
    def __init__(self, session_info, logger):
        self.session = session_info

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
                   lab="Tripodi Lab")

        print("Processing complete.")
        # print time taken in mimutes and seconds:
        print(f"Time taken: {round((time.perf_counter() - start_time) // 60)} minutes, {round((time.perf_counter() - start_time) % 60)} seconds")



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

        for i, pulse in enumerate(self.camera_pulses):
            self.pulse_times[i] = pulse

        if len(self.camera_pulses) < self.frame_IDs[-1]:
            print(f"Pulses: {len(self.camera_pulses)}, Frame IDs: {self.frame_IDs[-1]}")
            raise Exception("Error: Number of camera pulses is less than the number of frame IDs. Check data.")

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

            length_timestamps = len(self.timestamps)
            length_scales = len(scales_logs)

            new_scales_timestamps = []

            for i in range(0, length_timestamps, length_timestamps // length_scales):
                try:
                    new_scales_timestamps.append(self.timestamps[i])
                except IndexError:
                    new_scales_timestamps.append(self.timestamps[-1])

            # if length of timestamps is longer than number of weight readings, cut off last timestamps to make equal length:
            if len(new_scales_timestamps) > len(scales_logs):
                new_scales_timestamps = new_scales_timestamps[:len(scales_logs)]

            self.scales_data = {}
            self.scales_data["timestamps"] = new_scales_timestamps
            self.scales_data["weights"] = [value_pair[1] for value_pair in scales_logs]
            self.scales_data["sendkey timestamps"] = [value_pair[0] for value_pair in scales_logs]

            self.scales_data["mouse_weight_threshold"] = self.sendkey_logs.mouse_weight

            print("**Warning: Scales data not accurately timestamped.**")


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

    def bmp_to_avi_MP(self, prefix, framerate = 30, num_processes = 8):
        # Get all the bmp files in the folder
        bmp_files = [f for f in os.listdir(self.data_folder_path) if f.endswith('.bmp') and f.startswith(prefix)]

        # Sort the files by name
        bmp_files.sort()

        # Get the first file to use as a template for the video writer
        first_file = cv.imread(os.path.join(self.data_folder_path, bmp_files[0]))
        height, width, channels = first_file.shape
        self.dims = (width, height)
        self.FPS = framerate

        temp_video_dir = self.data_folder_path / 'temp_videos'
        os.makedirs(temp_video_dir, exist_ok=True)

        # Divide your list of bmp frame files into chunks according to the number of available CPUs
        chunk_size = len(bmp_files) // num_processes
        chunks = [bmp_files[i:i + chunk_size] for i in range(0, len(bmp_files), chunk_size)]

        # Use multiprocessing to process each chunk
        with mp.Pool(num_processes) as p:
            p.starmap(process_video_chunk_MP, [(chunks[i], i, temp_video_dir, self.FPS, self.dims, self.data_folder_path) for i in range(num_processes)])

        # Concatenate all chunks into a single video
        output_path = self.data_folder_path / f"{self.data_folder_path.stem}_{prefix}_MP.avi"
        self.concatenate_videos(temp_video_dir, output_path)

        # Clean up the temporary directory
        os.rmdir(temp_video_dir)
    
    def get_dims(self, frame_path):
        with open(frame_path, 'rb') as bmp:
            bmp.read(18)  # Skip over the size and reserved fields.

            # Read width and height.
            width = struct.unpack('I', bmp.read(4))[0]
            height = struct.unpack('I', bmp.read(4))[0]

        return (width, height)
    
    def concatenate_videos(self, temp_video_dir, output_path):
        # Determine the list of all chunk video files
        chunk_files = sorted([os.path.join(temp_video_dir, f) for f in os.listdir(temp_video_dir) if f.endswith('.avi')])
        # Create a temporary text file containing the list of video files for ffmpeg
        list_path = os.path.join(temp_video_dir, 'video_list.txt')
        with open(list_path, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")

        # Run ffmpeg command to concatenate all the videos
        ffmpeg_cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_path]
        subprocess.run(ffmpeg_cmd)

        # Clean up the temporary chunk video files and text file
        for file_path in chunk_files:
            os.remove(file_path)
        os.remove(list_path)

    def clear_BMP_files(self):
        # Get all the bmp files in the folder
        bmp_files = [f for f in os.listdir(self.path) if f.endswith('.bmp')]

        # Sort the files by name
        bmp_files.sort()

        for bmp_file in bmp_files:
            bmp_path = os.path.join(self.path, bmp_file)
            os.remove(bmp_path)

def bmp_to_avi_MP(prefix, data_folder_path, framerate = 30, num_processes = 8):
    # Get all the bmp files in the folder
    bmp_files = [f for f in os.listdir(data_folder_path) if f.endswith('.bmp') and f.startswith(prefix)]

    # Sort the files by name
    bmp_files.sort()

    # Get the first file to use as a template for the video writer
    first_file = cv.imread(os.path.join(data_folder_path, bmp_files[0]))
    height, width, channels = first_file.shape
    dims = (width, height)
    FPS = framerate

    temp_video_dir = data_folder_path / 'temp_videos'
    os.makedirs(temp_video_dir, exist_ok=True)

    # Divide your list of bmp frame files into chunks according to the number of available CPUs
    chunk_size = len(bmp_files) // num_processes
    chunks = [bmp_files[i:i + chunk_size] for i in range(0, len(bmp_files), chunk_size)]

    # Use multiprocessing to process each chunk
    with mp.Pool(num_processes) as p:
        p.starmap(process_video_chunk_MP, [(chunks[i], i, temp_video_dir, FPS, dims, data_folder_path) for i in range(num_processes)])

    # Concatenate all chunks into a single video
    output_path = data_folder_path / f"{data_folder_path.stem}_{prefix}_MP.avi"
    concatenate_videos(temp_video_dir, output_path)

    # Clean up the temporary directory
    os.rmdir(temp_video_dir)

def get_dims(frame_path):
    with open(frame_path, 'rb') as bmp:
        bmp.read(18)  # Skip over the size and reserved fields.

        # Read width and height.
        width = struct.unpack('I', bmp.read(4))[0]
        height = struct.unpack('I', bmp.read(4))[0]

    return (width, height)

def concatenate_videos(temp_video_dir, output_path):
    # Determine the list of all chunk video files
    chunk_files = sorted([os.path.join(temp_video_dir, f) for f in os.listdir(temp_video_dir) if f.endswith('.avi')])
    # Create a temporary text file containing the list of video files for ffmpeg
    list_path = os.path.join(temp_video_dir, 'video_list.txt')
    with open(list_path, 'w') as f:
        for chunk_file in chunk_files:
            f.write(f"file '{chunk_file}'\n")

    # Run ffmpeg command to concatenate all the videos
    ffmpeg_cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_path]
    subprocess.run(ffmpeg_cmd)

    # Clean up the temporary chunk video files and text file
    for file_path in chunk_files:
        os.remove(file_path)
    os.remove(list_path)

def clear_BMP_files(data_folder_path):
    # Get all the bmp files in the folder
    bmp_files = [f for f in os.listdir(data_folder_path) if f.endswith('.bmp')]

    # Sort the files by name
    bmp_files.sort()

    for bmp_file in bmp_files:
        bmp_path = os.path.join(data_folder_path, bmp_file)
        os.remove(bmp_path)

def process_video_chunk_MP(chunk, chunk_index, temp_video_dir, FPS, DIMS, path):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # Each process will create its own output file
    temp_video_path = os.path.join(temp_video_dir, f"chunk_{chunk_index}.avi")
    video_writer = cv.VideoWriter(temp_video_path, fourcc, FPS, DIMS)

    for bmp_file in chunk:
        bmp_path = os.path.join(path, bmp_file)
        frame = cv.imread(bmp_path)
        video_writer.write(frame)

    video_writer.release()




def main():

    cohort_directory = Path(r"/cephfs2/srogers/March_training")

    directory_info = Cohort_folder(cohort_directory, OEAB_legacy = False).cohort

    no_of_session_in_cohort = 0
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            no_of_session_in_cohort += 1

    analysis_log_filename = f"{cohort_directory}/{datetime.now():%y%m%d_%H%M%S}_analysis_log.txt"
    analysis_log = f"Analysis_manager run {datetime.now():%d-%m-%Y_%H:%M:%S}\n"
    with open(analysis_log_filename, 'w') as f:
        f.write(analysis_log)

    fully_processed = 0
    incomplete = 0

    refresh = False
    
    sessions_completed = 0
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            session_directory = Path(directory_info["mice"][mouse]["sessions"][session]["directory"])
            if directory_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                if not directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"] == True or refresh == True:
                    print(f"Processing {session_directory}...")
                    Process_Raw_Behaviour_Data(session_directory)
                    sessions_completed += 1
                    fully_processed += 1
                    print(f"{sessions_completed}/{no_of_session_in_cohort} completed")
                    analysis_log += f"{session_directory} processed\n"
                    with open(analysis_log_filename, 'w') as f:
                        f.write(analysis_log)
                else:
                    print(f"{session_directory} already processed")
                    analysis_log += f"{session_directory} already processed\n"  
                    with open(analysis_log_filename, 'w') as f:
                        f.write(analysis_log)
                    sessions_completed += 1
                    fully_processed += 1
                    print(f"{sessions_completed}/{no_of_session_in_cohort} completed")
            else:
                print(f"{session_directory} raw data incomplete")
                analysis_log += f"{session_directory} raw data incomplete\n"
                with open(analysis_log_filename, 'w') as f:
                    f.write(analysis_log)
                sessions_completed += 1
                incomplete += 1
                print(f"{sessions_completed}/{no_of_session_in_cohort} completed")

    

    print(f"Fully processed: {fully_processed}, incomplete: {incomplete}")

def main_MP():

    total_start_time = time.perf_counter()

    cohort_directory = Path(r"/cephfs2/srogers/Behaviour code/test_data")

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

    Cohort = Cohort_folder(cohort_directory, multi = True, plot=False, OEAB_legacy = False)

    directory_info = Cohort.cohort

    sessions_to_process = []
    num_sessions = 0

    refresh = False
    
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

    # # sessions_to_process = ['240909_140114_mtao89-1d']
    # sessions_to_process = ['240909_162750_mtao93-1b']
    # sessions_to_process = [Cohort.get_session(session) for session in sessions_to_process]

    for session in sessions_to_process:
        print(f"Processing {session.get('directory')}...")
        Process_Raw_Behaviour_Data(session, logger)

    directory_info = Cohort_folder(cohort_directory, multi = True, plot=True, OEAB_legacy = False).cohort

    # print total time taken in minutes and seconds, rounded to whole numbers
        
    print(f"Total time taken: {round((time.perf_counter() - total_start_time) // 60)} minutes, {round((time.perf_counter() - total_start_time) % 60)} seconds")

if __name__ == "__main__":
    main_MP()






