import bisect
import numpy as np
from pynwb import NWBHDF5IO

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