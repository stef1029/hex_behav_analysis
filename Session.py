from pathlib import Path
import json
import pandas as pd
import bisect
import cv2 as cv
from DAQ_plot import DAQ_plot
import math
import numpy as np

class Session:
    def __init__(self, session_dict):
        self.session_dict = session_dict
        # print("Loading session...")
        self.session_directory = Path(self.session_dict.get('directory'))

        self.session_ID = self.session_dict.get('session_id')
        
        self.find_data()

        self.load_data()

        self.trial_detector = DetectTrials(self.processed_DAQ_data, self.session_metadata["session_behaviour_phase"])

        self.trials = self.trial_detector.trials
        # trials are currently formatted as a list of trial objects, each containing this data:
        # eg: {'start': 2234722, 'end': 2238854, 'correct_port': 3, 'next_sensor_time': 2238850, 'next_sensor_ID': 3}

        # only if the key is a number, not words:
        self.timestamps = [float(timestamp) for timestamp in self.processed_DAQ_data["timestamps"]]

        self.trials = self.add_timestamps(self.trials)

        self.indexed_frametimes = self.frametime_to_index(self.video_frametimes)

        self.add_video_data_to_trials()

        self.add_angle_data()

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

            # -------- GET CUE PRESENTATION ANGLE FROM MOUSE HEADING: ------------------------

            port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class

            correct_port = trial["correct_port"]
            if correct_port == "audio-1":
                correct_port = 1
            port = int(correct_port) - 1
            cue_angle = port_angles[port]

            cue_presentation_angle = (cue_angle - theta_deg) % 360

            if cue_presentation_angle > 180:
                cue_presentation_angle -= 360
            elif cue_presentation_angle <= -180:
                cue_presentation_angle += 360


            # -------- RETURN DATA: ------------------------
            angle_data = {"bearing": theta_deg, 
                        "port_position": port_angles[port],
                        "midpoint": midpoint,
                        "cue_presentation_angle": cue_presentation_angle}

            return angle_data
        else:
            return None


    def draw_LEDs(self, start = 0, end = None):

        port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
    
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

    def find_data(self):
        self.processed_DAQ_data_path = Path(self.session_dict.get('processed_data', {}).get('processed_DAQ_data'))
        self.sendkey_logs_path = Path(self.session_dict.get('processed_data', {}).get('sendkey_logs'))
        self.video_frametimes_path = Path(self.session_dict.get('processed_data', {}).get('video_frametimes'))
        self.session_video = Path(self.session_dict.get('raw_data', {}).get('raw_video'))
        self.sendkey_metadata_path = Path(self.session_dict.get('raw_data', {}).get('behaviour_data'))
        self.DLC_coords_path = Path(self.session_dict.get('processed_data', {}).get('DLC', {}).get('coords_csv'))

    def load_data(self):
        with open(self.processed_DAQ_data_path, 'r') as f:
            self.processed_DAQ_data = json.load(f)
            self.scales_data = self.processed_DAQ_data["scales_data"]

        with open(self.sendkey_logs_path, 'r') as f:
            self.sendkey_logs_json = json.load(f)
            self.sendkey_logs_dataframe = pd.read_json(self.sendkey_logs_json, orient='table')
        
        with open(self.video_frametimes_path, 'r') as f:
            self.video_frametimes = json.load(f)
        
        with open(self.sendkey_metadata_path, 'r') as f:
            self.sendkey_metadata = json.load(f)

        self.DLC_coords = pd.read_csv(self.DLC_coords_path, header=[1, 2])
        
        self.session_metadata = {
        "mouse_ID": self.sendkey_metadata.get("Mouse ID", None),
        "session_behaviour_phase": self.sendkey_metadata.get("Behaviour phase", None),
        "max_trials": self.sendkey_metadata.get("Number of trials", None),
        "mouse_weight": self.sendkey_metadata.get("Mouse weight", None),
        "ports_active": self.sendkey_metadata.get("Port", None),
        "mouse_weight_threshold": self.sendkey_metadata.get("Mouse weight threshold", None),
        "session_start_time": self.sendkey_metadata.get("Date and time", None),
        "session_end_time": self.sendkey_metadata.get("End time", None)
        }


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

        for j, trial in enumerate(self.trials):
            start = trial["start"]
            end = trial["end"]

            video_frames = []
            # use bisect to find the frametimes from the indexed_frametimes dictionary
            keys = list(self.indexed_frametimes.keys())
            start_index = keys[bisect.bisect_left(keys, start)-1]
            if end != None:
                end_index = keys[bisect.bisect_left(keys, end)-1]
            else:
                end_index = keys[-1]

            # for each index in between the start and end, see if it's in the indexed_frametimes dictionary, and if it is, add it to the video_frames list.
            # bug happened here where I was appending the frame_ID not the frame number.
            # when the video processor later went to grab the appropriate frame, it grabbed the wrong frame, since it needed the index in the video.
            for i in range(start_index, end_index):
                if i in self.indexed_frametimes:
                    video_frames.append(self.indexed_frametimes[i]["frame_no"])
            
            trial["video_frames"] = video_frames

            # also add the DLC data for the same time period, if it's been found:
            if self.DLC_coords is not None:
                if len(video_frames) != 0:
                    try:
                        trial["DLC_data"] = self.DLC_coords.loc[video_frames[0]:video_frames[-1]]
                    except IndexError:
                        print(f"max dlc_coord = {len(self.DLC_coords)}")
                        print(f"max video_frames = {len(self.indexed_frametimes)}")
                        print(list(self.indexed_frametimes.keys())[-1])
                        print(f"num trials = {len(self.trials)}")
                        print(start, end)
                        raise Exception(f"error happening on trial {j}")
                else:
                    trial["DLC_data"] = None

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
        return a list of 6 angles, each representing the angle of a port from the platform centre.
        """
        # grab first frame from trial 1:
        frame_no = self.trials[0]["video_frames"][0]
        # this will be the image displayed for the user to click on.
        port_angles = [0, 60, 120, 180, 240, 300]

        calibration_offset = 64         # change this offset here to rotate the lines in the image around until they line up

        # open video and grab frame:
        cap = cv.VideoCapture(str(self.session_video))
        dims = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        # for each angle in port_angles, draw a line from the centre.
        for angle in port_angles:
            x2 = int(dims[0]/2 + 400 * math.cos(math.radians(angle + calibration_offset)))
            y2 = int(dims[1]/2 - 400 * math.sin(math.radians(angle + calibration_offset)))
            cv.line(frame, (int(dims[0]/2), int(dims[1]/2)), (x2, y2), (0, 255, 0), 2)

            # at end of line, write index of port
            cv.putText(frame, str(port_angles.index(angle)+1), (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # save frame to file
        filename = "test_output/angle_frame.jpg"        # may need to change to appropriate directory
        cv.imwrite(filename, frame)

        print([angle + calibration_offset for angle in port_angles])    # prints a list to copied to the main function



class DetectTrials:
    """
    Detects trials based on the phase thats sent to it.
    Most phases should come under the normal detector, with audio, except 1, 2 and potentially the wait trails.
    """
    def __init__(self, data, phase):
        self.data = data
        self.phase = phase

        self.timestamps = [float(timestamp) for timestamp in self.data["timestamps"]]

        if phase == "1" or phase == "2":
            self.early_phase_find_trials(1)
        else:
            self.trials = self.create_trial_list()

        # print(self.get_session_stats(trials))

    def create_trial_list(self):
        
        trial_list = []

        for i in range(1, 7):
            channel = f"LED_{i}"
            trials = self.find_trials(channel)
            for trial in trials:
                trial_list.append(trial)
        trials = self.find_trials("GO_CUE")
        for trial in trials:
            trial_list.append(trial)

        # sort trial list by start time:
        trial_list.sort(key=lambda trial: trial["start"])

        return trial_list

    def find_trials(self, channel):
        """
        Goes through each port one at a time finding trials.
        """
        trials = []
        # ------- Find events for this LED: ------------
        cues = []
        i = 0
        while True:

            event = self.find_next_event(channel, i, 1)

            event["correct_port"] = channel[-1] if channel != "GO_CUE" else "audio-1"

            if event["start"] != None:
                cues.append(event)

            if event["end"] != None:
                i = event["end"] + 1
            
            if event["end"] == None:
                break
        # ----------------------------------------------
        
        # For each event in the LED list, go through each of the sensor channels and find the next event, and make a list of these.
        for event in cues:
            sensor_events = []
            for i in range(1, 7):
                sensor_name = f"SENSOR{i}"
                sensor_events.append(self.find_next_event(sensor_name, event["start"], high_value = 0))

            index_value_pairs = [(i, events["start"]) for i, events in enumerate(sensor_events) if events["start"] != None]

            if index_value_pairs:  # Make sure the list is not empty
                min_index, min_start = min(index_value_pairs, key=lambda pair: pair[1])
                event["next_sensor_time"] = min_start
                event["next_sensor_ID"] = min_index + 1
            else:
                event["next_sensor_time"] = None
                event["next_sensor_ID"] = None
        # ----------------------------------------------
            trials.append(event)
        return trials
        
    def find_next_event(self, channel, start_index, high_value = 1):

        HIGH = f"{high_value}"
        if high_value == 1: LOW = "0" 
        else: LOW = "1"

        max_index = len(self.data[channel]) - 1

        start = None
        end = None

        event = {}

        i = start_index    
        while True:
            channel_state = self.data[channel][i]

            if channel_state == HIGH and self.data[channel][i-1] == LOW:        # Channel has just gone high
                start = i

            if channel_state == LOW and self.data[channel][i-1] == HIGH:        # Channel has just gone low
                end = i

            if start != None and end != None:
                break

            if i == max_index:
                break  

            i += 1

        event["start"] = start
        event["end"] = end

        return event
    

    def early_phase_find_trials(self, channel):
        """
        Find the first sensor touch on the correct port, then use that index to find all the other sensor touches before it. 
        If not found on the correct port, then a timeout occured and the 30 seconds of data is collected.
        """
        # ------- Find events for this valve: ------------
        # Finds a list of all the valve activations.

        port = channel
        channel = f"VALVE{channel}"

        self.trials = []
        solenoid_activations = []
        i = 0
        while True:

            event = self.find_next_event(channel, i, 1)

            event["correct_port"] = port

            if event["start"] != None:
                solenoid_activations.append(event)

            if event["end"] != None:
                i = event["end"] + 1
            
            if event["end"] == None:
                break

        # Now, using those valve activations as starting points, it first finds the index when the corresponding sensor got touched.
        # Then, it finds all the other sensor touches before that index.
            
        for event in solenoid_activations:
            sensor_1_touch = self.find_next_event(f"SENSOR{channel[-1]}", event["start"], high_value = 0)
            event["reward_touch"] = sensor_1_touch
            start_search = event["start"]
            end_search = sensor_1_touch["start"]

            if end_search != None:
                sensor_touches = []
                for i in range(1, 7):
                    sensor = f"SENSOR{i}"
                    i = start_search
                    while True:
                        new_event = self.find_next_event(sensor, i, high_value = 0)
                        if new_event["start"] != None and new_event["start"] < end_search:
                            new_event["port"] = sensor[-1]
                            sensor_touches.append(new_event)
                            if new_event["end"] != None:
                                i = new_event["end"] + 1
                            else:
                                break
                        else:
                            break
        
        # sort sensor touches by start index:
            sensor_touches.sort(key=lambda touch: touch["start"])

            event["sensor_touches"] = sensor_touches

            self.trials.append(event)
                    
                


    # def get_session_stats(self, trials):

    #     session_stats = {}

    #     session_stats["num_trials"] = len(trials)
    #     session_stats["num_successes"] = len([trial for trial in trials if trial["success"] == True])
    #     session_stats["num_failures"] = len([trial for trial in trials if trial["success"] == False])
    #     # total time in minutes and seconds, rounded:
    #     session_stats["total_time"] = f"{round((self.timestamps[-1] - self.timestamps[0]) / 60, 2)} minutes, {round((self.timestamps[-1] - self.timestamps[0]) % 60, 2)} seconds"
    #     session_stats["average_time_between_trials"] = f"{round((self.timestamps[-1] - self.timestamps[0]) / session_stats['num_trials'], 2)} seconds"
    #     session_stats["average_time_to_next_sensor"] = f"{round(sum([trial['time_to_next_sensor'] for trial in trials if trial['time_to_next_sensor'] != None]) / session_stats['num_trials'], 2)} seconds"
    #     session_stats["success_rate"] = session_stats["num_successes"] / session_stats["num_trials"]

    #     return session_stats


def main():
    test_dir = Path(r"/cephfs2/srogers/December_training_data/231220_115610_wtjx285-2d")

    session = Session(test_dir)



if __name__ == "__main__":
    main()


