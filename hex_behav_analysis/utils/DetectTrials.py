import bisect
import numpy as np
from pynwb import NWBHDF5IO
from functools import lru_cache

class DetectTrials:
    """
    Optimized version of DetectTrials with significant performance improvements.
    """
    def __init__(self, nwbfile_path):
        self.nwbfile_path = nwbfile_path
        
        # Load all data once and cache it
        self._load_all_data()
        
        # Extract metadata
        session_metadata = self.nwbfile_data['metadata']['experiment_description']
        self.phase = session_metadata.split(";")[0].split(":")[1]
        self.rig_id = session_metadata.split(";")[1].split(":")[1]
        self.last_timestamp = self.nwbfile_data['metadata']['last_timestamp']
        
        # Create trials
        self.trials = self.create_trial_list(phase=self.phase)
    
    def _load_all_data(self):
        """
        Load all required data from NWB file once and cache it in memory.
        This avoids repeated file I/O operations.
        """
        self.nwbfile_data = {
            'stimulus': {},
            'acquisition': {},
            'metadata': {}
        }
        
        with NWBHDF5IO(self.nwbfile_path, 'r') as io:
            nwbfile = io.read()
            
            # Cache metadata
            self.nwbfile_data['metadata']['experiment_description'] = nwbfile.experiment_description
            self.nwbfile_data['metadata']['last_timestamp'] = nwbfile.acquisition['scales'].timestamps[-1]
            
            # Cache all LED and GO_CUE data
            for i in range(1, 7):
                led_channel = f"LED_{i}"
                if led_channel in nwbfile.stimulus:
                    ts = nwbfile.stimulus[led_channel]
                    self.nwbfile_data['stimulus'][led_channel] = {
                        'data': ts.data[:],
                        'timestamps': ts.timestamps[:]
                    }
                
                valve_channel = f"VALVE{i}"
                if valve_channel in nwbfile.stimulus:
                    ts = nwbfile.stimulus[valve_channel]
                    self.nwbfile_data['stimulus'][valve_channel] = {
                        'data': ts.data[:],
                        'timestamps': ts.timestamps[:]
                    }
            
            # Cache GO_CUE
            if "GO_CUE" in nwbfile.stimulus:
                ts = nwbfile.stimulus["GO_CUE"]
                self.nwbfile_data['stimulus']["GO_CUE"] = {
                    'data': ts.data[:],
                    'timestamps': ts.timestamps[:]
                }
            
            # Cache all sensor data
            for i in range(1, 7):
                sensor_channel = f"SENSOR{i}"
                if sensor_channel in nwbfile.acquisition:
                    ts = nwbfile.acquisition[sensor_channel]
                    self.nwbfile_data['acquisition'][sensor_channel] = {
                        'data': ts.data[:],
                        'timestamps': ts.timestamps[:]
                    }

    def create_trial_list(self, phase):
        trial_list = []
        
        if phase not in ["1", "2"]:
            # Process LED channels
            for i in range(1, 7):
                channel = f"LED_{i}"
                trials = self.find_trials_cue(channel)
                for trial in trials:
                    trial_list.append(trial)
            
            # Process GO_CUE
            trials = self.find_trials_cue("GO_CUE")
            for trial in trials:
                trial_list.append(trial)
        else:
            # Phase 1 or 2: use valve data
            for i in range(1, 7):
                channel = f"VALVE{i}"
                trials = self.find_trials_cue(channel)
                for trial in trials:
                    trial_list.append(trial)
        
        # Sort once
        trial_list.sort(key=lambda trial: trial["cue_start"])
        
        # Find sensor touches for all trials
        trial_list = self.find_trials_sensor(trial_list, phase)
        
        # Add phase info
        for i, trial in enumerate(trial_list):
            trial["phase"] = phase
        
        # Handle special cases
        if phase == "9c":
            trial_list = self.check_go_cue_activation(trial_list)
        
        if phase == '10':
            trial_list = self.merge_trials(trial_list)
        
        # Final sort and numbering
        trial_list.sort(key=lambda trial: trial["cue_start"])
        for i, trial in enumerate(trial_list):
            trial['trial_no'] = i
        
        return trial_list
    
    def find_trials_cue(self, channel):
        """
        Optimized version using cached data and vectorized operations.
        """
        trials = []
        
        if channel not in self.nwbfile_data['stimulus']:
            return trials
        
        cue_data = self.nwbfile_data['stimulus'][channel]['data']
        cue_timestamps = self.nwbfile_data['stimulus'][channel]['timestamps']
        
        if len(cue_data) == 0:
            return trials
        
        if 'VALVE' not in channel:
            # Check for stuck pins
            unique_values = np.unique(cue_data)
            if len(unique_values) == 1 and unique_values[0] == 1:
                print(f"Warning: {channel} appears to be stuck high. Skipping.")
                return trials
            
            # Check for too few transitions
            transitions = np.diff(cue_data)
            num_transitions = np.sum(transitions != 0)
            if num_transitions < 4:
                print(f"Warning: {channel} has only {num_transitions} transitions. Skipping.")
                return trials
            
            # Determine starting point
            if cue_data[0] == 1:
                start = 0
            else:
                start = 1
            
            # Get indices for starts and ends
            start_indices = np.arange(start, len(cue_data), 2)
            end_indices = np.arange(start + 1, len(cue_data), 2)
            
            start_timestamps = cue_timestamps[start_indices]
            
            # Handle incomplete final event
            if cue_data[-1] == 1 and len(end_indices) < len(start_indices):
                end_timestamps = np.append(cue_timestamps[end_indices], None)
            else:
                end_timestamps = cue_timestamps[end_indices]
            
            # Get correct port
            correct_port = channel[-1] if channel != "GO_CUE" else "audio-1"
            
            # Create trials
            trials = [{'correct_port': correct_port, 'cue_start': start, 'cue_end': end}
                     for start, end in zip(start_timestamps, end_timestamps)]
            
        else:
            # VALVE handling
            trials = [{'correct_port': channel[-1], 
                      'cue_start': cue_timestamps[i], 
                      'cue_end': cue_timestamps[i] + cue_data[i]} 
                     for i in range(len(cue_timestamps))]
        
        return trials
    
    def find_trials_sensor(self, trials, phase):
        """
        Optimized sensor finding while maintaining original logic.
        """
        # Initialize sensor_touches for all trials
        for trial in trials:
            trial["sensor_touches"] = []
        
        # Process each sensor channel
        for i in range(1, 7):
            channel = f"SENSOR{i}"
            
            if channel not in self.nwbfile_data['acquisition']:
                continue
            
            sensor_data = self.nwbfile_data['acquisition'][channel]['data']
            sensor_timestamps = self.nwbfile_data['acquisition'][channel]['timestamps']
            
            # Process each trial
            for j, trial in enumerate(trials):
                if phase == '10':
                    start = trial['cue_end']
                else:
                    start = trial["cue_start"]
                
                # Find end time
                end = self.last_timestamp
                for k in range(j + 1, len(trials)):
                    if trials[k]['cue_start'] > trial['cue_start']:
                        end = trials[k]['cue_start']
                        break
                
                # Find sensor touches in time window
                start_index = bisect.bisect_left(sensor_timestamps, start)
                end_index = bisect.bisect_left(sensor_timestamps, end)
                
                # Create sensor touch entries
                sensor_touches = []
                for k in range(start_index, end_index):
                    if sensor_data[k] == 1:
                        sensor_touch = {
                            'sensor_touched': channel[-1],
                            'sensor_start': sensor_timestamps[k],
                            'sensor_end': sensor_timestamps[k+1] if k+1 < len(sensor_timestamps) else None
                        }
                        sensor_touches.append(sensor_touch)
                
                # Extend trial's sensor_touches list
                trial["sensor_touches"].extend(sensor_touches)
        
        # Sort sensor touches and find next sensor for each trial
        for trial in trials:
            # Sort by start time
            trial["sensor_touches"].sort(key=lambda touch: touch["sensor_start"])
            
            # Find next sensor
            trial["next_sensor"] = next(
                (sensor for sensor in trial["sensor_touches"] 
                 if sensor["sensor_start"] > trial["cue_start"]),
                {}
            )
            
            # Determine success
            if trial["next_sensor"] != {}:
                port_touched = trial["next_sensor"]["sensor_touched"]
                if trial["correct_port"] == "audio-1" and port_touched == "1":
                    trial["success"] = True
                else:
                    trial["success"] = trial["next_sensor"]["sensor_touched"] == trial["correct_port"]
            else:
                trial["success"] = False
        
        return trials
    
    def check_go_cue_activation(self, trials):
        """
        Check if GO_CUE was activated between LED cue and first sensor touch.
        """
        if "GO_CUE" not in self.nwbfile_data['stimulus']:
            for trial in trials:
                trial["go_cue"] = None
            return trials
        
        go_cue_data = self.nwbfile_data['stimulus']["GO_CUE"]['data']
        go_cue_timestamps = self.nwbfile_data['stimulus']["GO_CUE"]['timestamps']
        
        # Find all activation times
        go_cue_activations = go_cue_timestamps[go_cue_data == 1]
        
        for trial in trials:
            cue_start = trial["cue_start"]
            sensor_touches = trial["sensor_touches"]
            
            # Check if there are sensor touches
            first_sensor_touch_time = sensor_touches[0]["sensor_start"] if sensor_touches else False
            
            if not first_sensor_touch_time:
                trial["go_cue"] = None
            else:
                # Find GO_CUE between cue and touch
                go_cue_between = go_cue_activations[
                    (go_cue_activations > cue_start) & 
                    (go_cue_activations < first_sensor_touch_time)
                ]
                trial["go_cue"] = go_cue_between[0] if len(go_cue_between) > 0 else None
        
        return trials
    
    def merge_trials(self, trial_list):
        """
        Merge trials with same start time.
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
            
            # Check if trials should be merged
            if round(trial_1['cue_start'], 2) == round(trial_2['cue_start'], 2):
                if (trial_1['correct_port'] in {'1', '2', '3', '4', '5', '6'} and trial_2['correct_port'] == 'audio-1') or \
                   (trial_2['correct_port'] in {'1', '2', '3', '4', '5', '6'} and trial_1['correct_port'] == 'audio-1'):
                    
                    # Determine LED and audio trials
                    led_trial = trial_1 if trial_1['correct_port'] in {'1', '2', '3', '4', '5', '6'} else trial_2
                    audio_trial = trial_1 if trial_1['correct_port'] == 'audio-1' else trial_2
                    
                    # Update LED trial with audio trial info
                    led_trial['cue_end'] = audio_trial['cue_end']
                    led_trial['next_sensor'] = audio_trial['next_sensor']
                    led_trial['catch'] = True
                    
                    merged_trials.append(led_trial)
                    skip_next = True
                    
                    # Skip subsequent trials within range
                    j = i + 2
                    while j < len(trial_list):
                        if trial_list[j]['cue_start'] >= led_trial['cue_start'] and trial_list[j]['cue_start'] <= led_trial['cue_end']:
                            skip_next = True
                            j += 1
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
        
        # Handle last trial
        if not skip_next and i < len(trial_list):
            trial_list[i]['catch'] = False
            merged_trials.append(trial_list[i])
        
        return merged_trials