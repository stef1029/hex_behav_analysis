from pathlib import Path
import json
import pandas as pd
import bisect
import cv2 as cv
from DAQ_plot import DAQ_plot
import math
import numpy as np
from datetime import datetime
import traceback

from pynwb import NWBHDF5IO, NWBFile, TimeSeries, ProcessingModule, load_namespaces
from pynwb.behavior import SpatialSeries
from pynwb.file import Subject, LabMetaData
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, export_spec

from Cohort_folder import Cohort_folder

class Session:
    def __init__(self, session_dict):
        try:
            self.session_dict = session_dict


            
            self.session_directory = Path(self.session_dict.get('directory'))

            self.session_ID = self.session_dict.get('session_id')
            print(f"Loading session {self.session_ID}...")
            
            self.portable = self.session_dict['portable']

            if self.portable == True:
                self.nwb_file_path = Path(self.session_dict.get('NWB_file'))
            
            else:
                self.nwb_file_path = Path(self.session_dict.get('processed_data', {}).get('NWB_file'))
                self.DLC_coords_path = Path(self.session_dict.get('processed_data', {}).get('DLC', {}).get('coords_csv'))

            with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
                nwbfile = io.read()
                DLC_coords_present = 'behaviour_coords' in nwbfile.processing
            

            self.load_data()

            self.trial_detector = DetectTrials(self.nwb_file_path)

            self.trials = self.trial_detector.trials
            # print(self.trials[:10])

            # # trials are currently formatted as a list of trial objects, each containing this data:
            # # eg: {'start': 2234722, 'end': 2238854, 'correct_port': 3, 'next_sensor_time': 2238850, 'next_sensor_ID': 3}

            # # only if the key is a number, not words:
            # self.timestamps = [float(timestamp) for timestamp in self.processed_DAQ_data["timestamps"]]

            # self.trials = self.add_timestamps(self.trials)

            # self.indexed_frametimes = self.frametime_to_index(self.video_frametimes)

            self.add_video_data_to_trials()

            

            self.add_angle_data()
        except Exception:
            print(f"Error loading session {self.session_ID}")
            traceback.print_exc()
            pass
        # print(self.trials[0])

        # print("Session loaded.")

    def add_timestamps(self, trials):
        new_trials = []
        for trial in trials:
            new_trial = trial
            new_trial["start_time"] = self.timestamps[trial["start"]]
            try:
                new_trial["end_time"] = self.timestamps[trial["next_sensor_time"]]
            except (KeyError, TypeError):
                try:
                    new_trial["end_time"] = self.timestamps[trial['reward_touch']['end']]
                except:
                    new_trial["end_time"] = self.timestamps[-1]

            new_trials.append(new_trial)

        return new_trials

    def get_video(self, trial):
        """
        returns: (video object set to start frame, end frame)\n
        Should really be a function within a true trial object, but for now it's here.
        """
        # open video and grab frame:
        cap = cv.VideoCapture(str(self.session_video))
        cap.set(cv.CAP_PROP_POS_FRAMES, trial["video_frames"][0])

        return cap, trial["video_frames"][-1]
        

    def add_angle_data(self):
        for trial in self.trials:
            trial["turn_data"] = self.find_angles(trial)


    def find_angles(self, trial, buffer = 1):
        """
        returns: for each trial, the cue angle from mouse heading
        """
        #  ----------------  GET MOUSE HEADING FROM DLC COORDS: --------------------
        if trial["DLC_data"] is not None:
            # to get mouse heading, take the ear coords, find angle between that and the nose coords, and then add 90 degrees to that angle.
            left_ear_coords = [(trial["DLC_data"]["left_ear"]["x"].iloc[i], trial["DLC_data"]["left_ear"]["y"].iloc[i]) for i in range(buffer)]
            average_left_ear = (sum(coord[0] for coord in left_ear_coords)/len(left_ear_coords), sum(coord[1] for coord in left_ear_coords)/len(left_ear_coords))

            right_ear_coords = [(trial["DLC_data"]["right_ear"]["x"].iloc[i], trial["DLC_data"]["right_ear"]["y"].iloc[i]) for i in range(buffer)]
            average_right_ear = (sum(coord[0] for coord in right_ear_coords)/len(right_ear_coords), sum(coord[1] for coord in right_ear_coords)/len(right_ear_coords))

            vector_x = average_right_ear[0] - average_left_ear[0]
            vector_y = average_right_ear[1] - average_left_ear[1]

            # Calculate the angle relative to the positive x-axis
            theta_rad = math.atan2(-vector_y, vector_x)
            theta_deg = math.degrees(theta_rad)
            theta_deg = (theta_deg + 90) % 360

            # Calculating the midpoint
            midpoint_x = (average_left_ear[0] + average_right_ear[0]) / 2
            midpoint_y = (average_left_ear[1] + average_right_ear[1]) / 2

            # Midpoint coordinates
            midpoint = (midpoint_x, midpoint_y)
            
            theta_rad = math.radians(theta_deg)
            eyes_offset = 40
            # Calculate the directional offsets using cosine and sine
            offset_x = eyes_offset * math.cos(theta_rad)  # Offset along x based on heading
            offset_y = eyes_offset * math.sin(theta_rad)  # Offset along y based on heading

            # New midpoint coordinates after applying the offset
            new_midpoint_x = midpoint_x + offset_x
            new_midpoint_y = midpoint_y - offset_y  # Subtract because y-coordinates increase downwards in image coordinates

            # New midpoint
            midpoint = (new_midpoint_x, new_midpoint_y)

            # -------- GET CUE PRESENTATION ANGLE FROM MOUSE HEADING: ------------------------

            # port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
            if self.rig_id == 1:
                self.port_angles = [64, 124, 184, 244, 304, 364] 
            elif self.rig_id == 2:
                self.port_angles = [240, 300, 360, 420, 480, 540]
            else:
                raise ValueError(f"Invalid rig ID: {self.rig_id}")

            # if the center is frame height/2, width/2, and the angle is the value in port_angles,
            # then the port coordinates are
            frame_height = 1080
            frame_width = 1280
            center_x = frame_width / 2
            center_y = frame_height / 2
            distance = (frame_height / 2) * 0.9         # offset from center to find coords

            self.port_coordinates = []
            for angle_deg in self.port_angles:
                # Convert angle from degrees to radians
                angle_rad = np.deg2rad(angle_deg)
                
                # Calculate coordinates
                x = int(center_x + distance * np.cos(angle_rad))
                y = int(center_y - distance * np.sin(angle_rad))  # Subtracting to invert y-axis direction
                
                # Append tuple of (x, y) to the list of coordinates
                self.port_coordinates.append((x, y))

            self.relative_angles = []
            # Convert mouse heading to radians for calculation
            mouse_heading_rad = np.deg2rad(theta_deg)

            for port_x, port_y in self.port_coordinates:
                # Calculate vector from midpoint to the port
                vector_x = port_x - midpoint[0]
                vector_y = port_y - midpoint[1]

                # Calculate the angle from the x-axis to this vector
                port_angle_rad = math.atan2(-vector_y, vector_x)

                # Calculate the relative angle
                relative_angle_rad = port_angle_rad - mouse_heading_rad

                # Convert relative angle to degrees and make sure it is within [0, 360)
                relative_angle_deg = math.degrees(relative_angle_rad) % 360

                # Append calculated relative angle to list
                self.relative_angles.append(relative_angle_deg)

            correct_port = trial["correct_port"]
            if correct_port == "audio-1":
                correct_port = 1
            port = int(correct_port) - 1
            cue_presentation_angle = self.relative_angles[port] % 360

            port_touched_angle = None
            if trial['next_sensor'] != {}:
                port_touched = trial['next_sensor'].get('sensor_touched')
                
                if port_touched != None:
                    port_touched_angle = self.relative_angles[int(port_touched) - 1] % 360
                    # print(port_touched_angle)
            if cue_presentation_angle > 180:
                cue_presentation_angle -= 360
            elif cue_presentation_angle <= -180:
                cue_presentation_angle += 360

            if port_touched_angle != None:
                if port_touched_angle > 180:
                    port_touched_angle -= 360
                elif port_touched_angle <= -180:
                    port_touched_angle += 360


            # -------- RETURN DATA: ------------------------
            angle_data = {"bearing": theta_deg, 
                        "port_position": self.port_coordinates[port],
                        "midpoint": midpoint,
                        "cue_presentation_angle": cue_presentation_angle,
                        "port_touched_angle": port_touched_angle}

            return angle_data
        else:
            return None

        
    def draw_LEDs(self, start = 0, end = None):
        
        if self.rig_id == 1:
            port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
        elif self.rig_id == 2:
            port_angles = [240, 300, 360, 420, 480, 540]
        else:
            raise ValueError(f"Invalid rig ID: {self.rig_id}")
    
        # take list of trials, and iterate through making a frametext dict for the video.
        frame_text = {}
        for i, trial in enumerate(self.trials):
            for frame in trial["video_frames"]:
                frame_text[frame] = {}
                frame_text[frame]["text"] = f"trial {i+1}, cue: {trial['correct_port']}"
                frame_text[frame]["cue"] = trial['correct_port']

        output_filename = "/cephfs2/srogers/test_output/LEDs.mp4"
        cap = cv.VideoCapture(str(self.session_video))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        centre = (int(frame_width/2), int(frame_height/2))
        cue_position = (frame_height / 2) - 25

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if start > frame_count:
            raise Exception(f"start frame {start} is greater than frame count {frame_count}")
        if end is None:
            end = frame_count
        end = end if end < frame_count else frame_count

        for i in range(start, end):
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                if i in frame_text:
                    frame = cv.putText(frame, frame_text[i]["text"], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    port = frame_text[i]["cue"]
                    try:
                        port = int(port)-1
                        port_angle = port_angles[port]
                        x2 = int(centre[0] + cue_position * math.cos(math.radians(port_angle)))
                        y2 = int(centre[1] - cue_position * math.sin(math.radians(port_angle)))
                        # draw a * at the cue position
                        frame = cv.drawMarker(frame, (x2, y2), (0, 255, 0), markerType=cv.MARKER_STAR, markerSize=30, thickness=2, line_type=cv.LINE_AA)
                    except ValueError:
                        frame = cv.putText(frame, "audio", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                out.write(frame)
        
        cap.release()
        out.release()


    def load_data(self):
        # Load the NWB file:
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            nwbfile = io.read()
            DLC_coords_present = 'behaviour_coords' in nwbfile.processing
            self.last_timestamp = nwbfile.acquisition['scales'].timestamps[-1]
            session_metadata = nwbfile.experiment_description
            
            # session metadata looks like this: "phase:x;rig;y". Extract x and y:
            self.phase = str(session_metadata.split(";")[0].split(":")[1])
            self.rig_id = int(session_metadata.split(";")[1].split(":")[1])

        # check if dlc coords are in nwb file already:
        if not DLC_coords_present:
            self.DLC_coords = pd.read_csv(self.DLC_coords_path, header=[1, 2], index_col=0)
            # if the behaviour coordinates data hasn't been added yet, add the data to the nwb file:
            self.add_DLC_coords_to_nwb()

    def add_DLC_coords_to_nwb(self):
        # Load the DLC coords:
        self.DLC_coords = pd.read_csv(self.DLC_coords_path, header=[1, 2], index_col=0)

        with NWBHDF5IO(str(self.nwb_file_path), 'a') as io:
            nwbfile = io.read()

            video_data_name = 'behaviour_video'
            if video_data_name in nwbfile.acquisition:
                video_data = nwbfile.acquisition[video_data_name]
                video_timestamps = video_data.timestamps[:]

                # Create a new processing module for the DLC coords:
                behaviour_module = ProcessingModule(name='behaviour_coords', description='DLC coordinates for behaviour')
                nwbfile.add_processing_module(behaviour_module)

                for body_part in self.DLC_coords.columns.levels[0]:
                    # Assuming DLC_coords is a pandas DataFrame
                    data = self.DLC_coords[body_part].values  # Shape should be [num_frames, 3]

                    if len(video_timestamps) > len(data):        
                        # Assuming that the first and last frames are the ones excluded
                        adjusted_timestamps = video_timestamps[:len(data)]
                    else:
                        adjusted_timestamps = video_timestamps

                    ts = TimeSeries(name=body_part,
                                    data=data,
                                    unit='pixels',
                                    timestamps=adjusted_timestamps,
                                    description=f"Coordinates and likelihood for {body_part}")

                    behaviour_module.add_data_interface(ts)

                io.write(nwbfile)
        

    def frametime_to_index(self, frametimes):
        # Initialize the dictionary for fast lookup
        index_frametimes = {}


        for i, (frame_ID, time) in enumerate(frametimes.items()):
            if self.is_number(frame_ID):
                # Use bisect to find the insertion index
                index = bisect.bisect_left(self.timestamps, float(time))

                # Check that the index is within the range of self.timestamps
                if index < len(self.timestamps):
                    # If the timestamp at the derived index is greater than the desired time,
                    # and we're not looking at the first element,
                    # move the index back to align with the "less than or equal" requirement
                    if index > 0 and self.timestamps[index] > time:
                        index -= 1

                # Create a small dictionary for the frame information and assign it to the index key
                frame_info = {'frame_ID': frame_ID, 'time': time, "frame_no": i}
                index_frametimes[index] = frame_info

        # print(f"Max index in frametimes: {max(index_frametimes.keys())}")
        # print(f"Max index in timestamps: {len(self.timestamps)}")

        return index_frametimes

    def add_video_data_to_trials(self):
        # get each trial object and attach a list of the relevant video frames between the start and end of the trial,
        # and also a dataframe of the DLC data for the same time period, using those frame times as the indeices for the slice.
        with NWBHDF5IO(self.nwb_file_path, 'r') as io:
            nwbfile = io.read()
            self.session_video_object = nwbfile.acquisition['behaviour_video']
            self.session_video = self.session_directory / Path(self.session_video_object.external_file[0])
            self.video_timestamps = self.session_video_object.timestamps[:]
            for j, trial in enumerate(self.trials):
                if trial["phase"] != "1" or trial["phase"] != "2":
                    start = trial["cue_start"]
                    if trial["next_sensor"].get("sensor_start") != None:
                        end = trial["next_sensor"]["sensor_start"]
                    else:
                        end = self.trials[j+1]["cue_start"] if j+1 < len(self.trials) else self.last_timestamp

                if trial["phase"] == "1" or trial["phase"] == "2":
                    start = trial["cue_start"]
                    end = self.trials[j+1]["cue_start"] if j+1 < len(self.trials) else self.last_timestamp

                video_frames = []
                

                # use bisect to find the frametimes from the indexed_frametimes dictionary

                start_index = np.searchsorted(self.video_timestamps, start, side='left')
                if end != None:
                    end_index = np.searchsorted(self.video_timestamps, end, side='left')
                else:
                    end_index = self.video_timestamps[-1]

                # for each index in between the start and end, see if it's in the indexed_frametimes dictionary, and if it is, add it to the video_frames list.
                # bug happened here where I was appending the frame_ID not the frame number.
                # when the video processor later went to grab the appropriate frame, it grabbed the wrong frame, since it needed the index in the video.
                for i in range(start_index, end_index):
                    video_frames.append(i)
                


                if len(video_frames) != 0:
                    if 'behaviour_coords' in nwbfile.processing:
                        behaviour_module = nwbfile.processing['behaviour_coords']

                        # Initialize an empty dictionary to store DLC data for the trial
                        trial_DLC_data = {}

                        for time_series_name in behaviour_module.data_interfaces:
                            ts = behaviour_module.data_interfaces[time_series_name]

                            # Get the timestamps and data for the body part
                            timestamps = ts.timestamps[:]
                            data = ts.data[:]
                            # print(data)

                            # Find indices for the trial time range
                            start_index = np.searchsorted(timestamps, start, side='left')
                            end_index = np.searchsorted(timestamps, end, side='left')

                            # Slice the data and timestamps for the trial
                            trial_data = data[start_index:end_index]
                            trial_timestamps = timestamps[start_index:end_index]

                            trial_DLC_data[time_series_name] = trial_data
                        
                        # Initialize a dictionary to hold the data for DataFrame construction
                        dlc_data_dict = {}

                        for body_part, data in trial_DLC_data.items():
                            # Assuming 'data' is a 2D array with columns for x, y, and confidence
                            # print(data)
                            x, y, likelihood = data.transpose()  # Transpose to separate the columns

                            # Populate the dictionary for DataFrame creation
                            dlc_data_dict[(body_part, 'x')] = x
                            dlc_data_dict[(body_part, 'y')] = y
                            dlc_data_dict[(body_part, 'likelihood')] = likelihood

                        if len(video_frames) > data.shape[0]:
                            video_frames = video_frames[:data.shape[0]]

                        # Create a MultiIndex DataFrame from the dictionary
                        dlc_df = pd.DataFrame(dlc_data_dict, index=video_frames)

                        # # Convert the dictionary to a DataFrame
                        # dlc_df = pd.DataFrame(trial_DLC_data, index=trial_timestamps)
                        # dlc_df.index.name = 'timestamps'

                        dlc_df['timestamps'] = trial_timestamps
                        trial["video_frames"] = video_frames
                        trial["DLC_data"] = dlc_df
                else:
                    trial["video_frames"] = video_frames
                    trial["DLC_data"] = None

                # also add the DLC data for the same time period, if it's been found:
                # if self.DLC_coords is not None:
                #     if len(video_frames) != 0:
                #         try:
                #             trial["DLC_data"] = self.DLC_coords.loc[video_frames[0]:video_frames[-1]]
                #         except IndexError:
                #             print(f"max dlc_coord = {len(self.DLC_coords)}")
                #             print(f"max video_frames = {len(self.indexed_frametimes)}")
                #             print(list(self.indexed_frametimes.keys())[-1])
                #             print(f"num trials = {len(self.trials)}")
                #             print(start, end)
                #             raise Exception(f"error happening on trial {j}")
                #     else:
                #         trial["DLC_data"] = None

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

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
    def calibrate_port_angles(self):
        """
        Return a list of 6 angles, each representing the angle of a port from the platform centre.
        """
        # grab first frame from trial 1:
        frame_no = self.trials[0]["video_frames"][0]
        port_angles = [0, 60, 120, 180, 240, 300]

        calibration_offset = 240  # change this offset here to rotate the lines in the image around until they line up

        # open video and grab frame:
        cap = cv.VideoCapture(str(self.session_video))
        if not cap.isOpened():
            print(f"Error opening video file {self.session_video}")
            return

        dims = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to grab frame at position {frame_no}")
            cap.release()
            return

        # for each angle in port_angles, draw a line from the centre.
        for angle in port_angles:
            x2 = int(dims[0]/2 + 400 * math.cos(math.radians(angle + calibration_offset)))
            y2 = int(dims[1]/2 - 400 * math.sin(math.radians(angle + calibration_offset)))
            cv.line(frame, (int(dims[0]/2), int(dims[1]/2)), (x2, y2), (0, 255, 0), 2)

            # at end of line, write index of port
            cv.putText(frame, str(port_angles.index(angle)+1), (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # save frame to file
        filename = "test_output/angle_frame.jpg"  # Ensure the directory exists
        cv.imwrite(filename, frame)

        cap.release()  # release the video capture object
        print([angle + calibration_offset for angle in port_angles])  # prints a list to be copied to the main function



class DetectTrials:
    """
    Detects trials based on the phase thats sent to it.
    Most phases should come under the normal detector, with audio, except 1, 2 and potentially the wait trails.
    """
    def __init__(self, nwbfile_path):
        self.nwbfile_path = nwbfile_path
        with NWBHDF5IO(self.nwbfile_path, 'r') as io:
            nwbfile = io.read()
            self.last_timestamp = nwbfile.acquisition['scales'].timestamps[-1]
            session_metadata = nwbfile.experiment_description
            
            # session metadata looks like this: "phase:x;rig;y". Extract x and y:
            self.phase = session_metadata.split(";")[0].split(":")[1]
            self.rig_id = session_metadata.split(";")[1].split(":")[1]
            # self.wait_duration = session_metadata.split(";")[2].split(":")[1]
            # self.cue_duration = session_metadata.split(";")[3].split(":")[1]

        # print(f"Phase: {phase}, Rig: {rig_id}")

        # self.phase = '9'  # for now, just set to 9 for testing
        # self.rig_id = '1'  # for now, just set to 1 for testing

        # if phase == "1" or phase == "2":
        #     self.early_phase_find_trials(1)
        # else:
        self.trials = self.create_trial_list(phase = self.phase)

        # print(self.get_session_stats(trials))

    def create_trial_list(self, phase):
        
        trial_list = []

        if phase not in ["1", "2"]:
            for i in range(1, 7):
                channel = f"LED_{i}"
                trials = self.find_trials_cue(channel)
                for trial in trials:
                    trial_list.append(trial)
            trials = self.find_trials_cue("GO_CUE")
            for trial in trials:
                trial_list.append(trial)
        


        if phase == "1" or phase == "2":        # if phase 1 or 2, use the valve data to find trials.
            for i in range(1, 7):
                channel = f"VALVE{i}"
                trials = self.find_trials_cue(channel)
                for trial in trials:
                    trial_list.append(trial)

        # sort trial list by start time:
        trial_list.sort(key=lambda trial: trial["cue_start"])

        trial_list = self.find_trials_sensor(trial_list, phase)
        for i, trial in enumerate(trial_list):
            trial["phase"] = phase
            # print(phase, i, trial['cue_start'])

        if phase == "9c":
            trial_list = self.check_go_cue_activation(trial_list)

        if phase == '10':
            trial_list = self.merge_trials(trial_list)
        
        trial_list.sort(key=lambda trial: trial["cue_start"])
        for i, trial in enumerate(trial_list):
            trial['trial_no'] = i

        return trial_list
    
    
    def merge_trials(self, trial_list):
        """
        Merge trials in the list based on matching cue_start and handle PWM dimming.
        
        Args:
            trial_list (list): List of trial dictionaries, each containing 'cue_start', 'cue_end', and 'correct_port' keys.
            
        Returns:
            list: Updated list of trials with merged trials.
        """
        merged_trials = []
        skip_next = False

        i = 0
        while i < len(trial_list) - 1:
            if skip_next:
                skip_next = False
                i += 1
                continue

            trial_1 = trial_list[i]
            trial_2 = trial_list[i + 1]

            # Compare the cue_start times to 2 decimal places
            if round(trial_1['cue_start'], 2) == round(trial_2['cue_start'], 2):
                if (trial_1['correct_port'] in {'1', '2', '3', '4', '5', '6'} and trial_2['correct_port'] == 'audio-1') or \
                (trial_2['correct_port'] in {'1', '2', '3', '4', '5', '6'} and trial_1['correct_port'] == 'audio-1'):
                    
                    # Determine the merged trial, and update the cue_end using the audio-1 trial
                    led_trial = trial_1 if trial_1['correct_port'] in {'1', '2', '3', '4', '5', '6'} else trial_2
                    audio_trial = trial_1 if trial_1['correct_port'] == 'audio-1' else trial_2
                    
                    # Update the cue_end of the LED trial to match the audio trial
                    led_trial['cue_end'] = audio_trial['cue_end']
                    led_trial['next_sensor'] = audio_trial['next_sensor']
                    
                    # Mark the trial as a catch trial
                    led_trial['catch'] = True
                    
                    # Remove any LED trials that fall within the time range of this catch trial
                    merged_trials.append(led_trial)
                    skip_next = True
                    
                    # Skip all trials with cue_start within the new led_trial['cue_start'] and led_trial['cue_end'] range
                    for j in range(i + 2, len(trial_list)):
                        if trial_list[j]['cue_start'] >= led_trial['cue_start'] and trial_list[j]['cue_start'] <= led_trial['cue_end']:
                            skip_next = True
                        else:
                            break
                    i = j
                else:
                    trial_1['catch'] = False
                    merged_trials.append(trial_1)
                    i += 1
            else:
                trial_1['catch'] = False
                merged_trials.append(trial_1)
                i += 1
    
        # Append the last trial if it wasn't merged
        if not skip_next and i < len(trial_list):
            trial_list[i]['catch'] = False
            merged_trials.append(trial_list[i])

        return merged_trials

    def find_trials_cue(self, channel):
        """
        Goes through each port one at a time finding trials.
        """
        trials = []
        # ------- Find events for cue: ------------
        with NWBHDF5IO(self.nwbfile_path, 'r') as io:
            nwbfile = io.read()

            channel_timeseries = nwbfile.stimulus[channel]      # grab whole timeseries for channel

            cue_data = channel_timeseries.data[:]               # grab data (array of [1 -1 1 -1 etc...])       If using VALVEs as cues, this is a list of durations.
            cue_timestamps = channel_timeseries.timestamps[:]   # grab timestamps corresponding to each of these points

            if 'VALVE' not in channel:
                if len(cue_data) > 0:
                    if cue_data[0] == 1:        # This check that it start from an LED going on during the session.
                        start = 0               # Needed in case an LED was on before a session started for some reason.
                    if cue_data[0] == -1:   
                        start = 1

                    # Get the indices for starts and ends of LED events
                    start_indices = np.arange(start, len(cue_data), 2)
                    end_indices = np.arange(start + 1, len(cue_data), 2)

                    start_timestamps = cue_timestamps[start_indices]  # Finds the corresponding timestamps for all led on times.

                    # Handle the case where the last LED event doesn't end within the data
                    if cue_data[-1] == 1 and len(end_indices) < len(start_indices):  # If the last data point is 'on', add a placeholder
                        end_timestamps = np.append(cue_timestamps[end_indices], None)  # Append None for the last unfinished event
                    else:
                        end_timestamps = cue_timestamps[end_indices]

                    correct_port = channel[-1] if channel != "GO_CUE" else "audio-1"  # Handle the channel naming

                    # Combine start and end timestamps into events
                    trials = [{'correct_port': correct_port, 'cue_start': start, 'cue_end': end}
                            for start, end in zip(start_timestamps, end_timestamps)]
                    
            if 'VALVE' in channel:
                # VALVE data stored differently, so to find start and end take the start timestamp and add the value that's in data.
                trials = [{'correct_port': channel[-1], 'cue_start': cue_timestamps[i], 'cue_end': cue_timestamps[i] + cue_data[i]} for i in range(0, len(cue_timestamps))]

            return trials
        # ----------------------------------------------
        # Now I have all the cue events from the channels. Now I need to find correponding sensor touches within the time between the start of the cue and the start of the next one.
        # For each event in the LED list, go through each of the sensor channels and find the next event, and make a list of these.
    def find_trials_sensor(self, trials, phase):
        with NWBHDF5IO(self.nwbfile_path, 'r') as io:
            nwbfile = io.read()
            for i in range(1, 7):
                channel = f"SENSOR{i}"

                sensor_timeseries = nwbfile.acquisition[channel]

                sensor_data = sensor_timeseries.data[:]
                sensor_timestamps = sensor_timeseries.timestamps[:]

                sensor_touches = []

                for j, trial in enumerate(trials):
                    if phase == '10':
                        start = trial['cue_end']    # start looking at the end of the trial because there will be a gap and this means the audio trials
                                                    #    will see the sensors after them rather than the being blocked by the flickering led flase trials. 
                    else:
                        start = trial["cue_start"]
                    # Find the next trial that starts after the current trial's cue_end
                    end = self.last_timestamp  # Default to last timestamp if no subsequent trial meets the criteria

                    for k in range(j + 1, len(trials)):
                        if trials[k]['cue_start'] > trial['cue_end']:
                            end = trials[k]['cue_start']
                            break
                        
                    # if trial['correct_port'] == 'audio-1':
                    # print(f"Checking timespan: {end - start}")
                    start_index = bisect.bisect_left(sensor_timestamps, start)
                    end_index = bisect.bisect_left(sensor_timestamps, end)
                    
                    sensor_touches = [{'sensor_touched': channel[-1], 
                                       'sensor_start': sensor_timestamps[k], 
                                       'sensor_end': sensor_timestamps[k+1] if k+1 < len(sensor_timestamps) else None} 
                                       for k in range(start_index, end_index) if sensor_data[k] == 1]              # error with sensor timestamps fixed here, if getting very long sensor activation, check the signs of the values
                                                                                                                #      to check you're looking for the activation times.

                    # check if 'sensor touches' key not in trial:
                    if "sensor_touches" not in trial:
                        trial[f"sensor_touches"] = sensor_touches
                    else:
                        trial[f"sensor_touches"].extend(sensor_touches)
                
            # sort sensor touches in each trial by sensor start time:
            for trial in trials:
                trial["sensor_touches"].sort(key=lambda touch: touch["sensor_start"])
                trial["next_sensor"] = trial["sensor_touches"][0] if len(trial["sensor_touches"]) > 0 else {}
                if trial["next_sensor"] != {}:
                    trial["success"] = True if trial["next_sensor"]["sensor_touched"][-1] == trial["correct_port"] else False
                else:
                    trial["success"] = False
           

            return trials
        
    def check_go_cue_activation(self, trials):
        """
        Check if GO_CUE was activated between the LED cue and the first sensor touch in each trial.
        Add a new key "go_cue" to the trial dictionary.
        """
        with NWBHDF5IO(self.nwbfile_path, 'r') as io:
            nwbfile = io.read()

            go_cue_timeseries = nwbfile.stimulus["GO_CUE"]
            go_cue_data = go_cue_timeseries.data[:]
            go_cue_timestamps = go_cue_timeseries.timestamps[:]

            # Find all GO_CUE activation times
            go_cue_activations = go_cue_timestamps[go_cue_data == 1]
            # print(go_cue_activations)

            for trial in trials:
                cue_start = trial["cue_start"]
                sensor_touches = trial["sensor_touches"]
                first_sensor_touch_time = sensor_touches[0]["sensor_start"] if sensor_touches else False

                if not first_sensor_touch_time:
                    trial["go_cue"] = None
                else:
                    # Check if there is any GO_CUE activation between the LED cue and the first sensor touch
                    go_cue_between = go_cue_activations[(go_cue_activations > cue_start) & (go_cue_activations < first_sensor_touch_time)]
                    trial["go_cue"] = go_cue_between[0] if len(go_cue_between) > 0 else None

        return trials

def main():

    cohort = Cohort_folder(r"/cephfs2/srogers/March_training", multi=True)

    test_dir = cohort.get_session("240327_151415")
    # test_dir = cohort.get_session("240323_164315")

    session = Session(test_dir)

    # session.calibrate_port_angles()



if __name__ == "__main__":
    main()


