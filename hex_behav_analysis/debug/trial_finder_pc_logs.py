"""
Simple LED_4 trial detector for validation purposes.
Detects trials using only DAQ traces from LED_4 channel.
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
import bisect
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
import json
import re


class LED4TrialDetector:
    """
    Detect trials using only LED_4 DAQ traces for validation.
    """
    
    def __init__(self, session_dict):
        """
        Initialise the detector with session information.
        
        Args:
            session_dict (dict): Session dictionary from Cohort_folder
        """
        self.session_dict = session_dict
        self.session_directory = Path(session_dict.get('directory'))
        self.session_id = session_dict.get('session_id')
        
        # Load NWB file path
        if session_dict.get('portable'):
            self.nwb_file_path = Path(session_dict.get('NWB_file'))
        else:
            self.nwb_file_path = Path(session_dict.get('processed_data', {}).get('NWB_file'))
        
        # Data containers
        self.led4_data = None
        self.sensor_data = {}
        self.valve_data = {}
        self.scales_data = None  # Add scales data container
        self.trials = []
        self.pc_log_trials = []  # For PC log-based detection
        self.all_pc_trials = []  # For all PC trials across all ports
        
        # Load behaviour data file path for PC logs
        self.behaviour_data_path = Path(session_dict.get('raw_data', {}).get('behaviour_data'))
        
    def load_daq_data(self):
        """
        Load LED_4, sensor, valve, and scales data from NWB file.
        """
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            nwbfile = io.read()
            
            # Get session metadata
            session_metadata = nwbfile.experiment_description
            self.phase = session_metadata.split(";")[0].split(":")[1]
            self.last_timestamp = nwbfile.acquisition['scales'].timestamps[-1]
            
            # Load LED_4 data
            if "LED_4" in nwbfile.stimulus:
                ts = nwbfile.stimulus["LED_4"]
                self.led4_data = {
                    'data': ts.data[:],
                    'timestamps': ts.timestamps[:]
                }
                print(f"Loaded LED_4 data: {len(self.led4_data['data'])} samples")
            else:
                raise ValueError("LED_4 channel not found in NWB file")
            
            # Load all sensor data
            for i in range(1, 7):
                sensor_channel = f"SENSOR{i}"
                if sensor_channel in nwbfile.acquisition:
                    ts = nwbfile.acquisition[sensor_channel]
                    self.sensor_data[i] = {
                        'data': ts.data[:],
                        'timestamps': ts.timestamps[:]
                    }
            
            # Load valve data for port 4
            if "VALVE4" in nwbfile.stimulus:
                ts = nwbfile.stimulus["VALVE4"]
                self.valve_data[4] = {
                    'data': ts.data[:],
                    'timestamps': ts.timestamps[:]
                }
                print(f"Loaded VALVE4 data: {len(self.valve_data[4]['timestamps'])} actuations")
            
            # Load scales data
            if "scales" in nwbfile.acquisition:
                scales_ts = nwbfile.acquisition["scales"]
                self.scales_data = {
                    'data': scales_ts.data[:],  # Weight in grams
                    'timestamps': scales_ts.timestamps[:]
                }
                # Extract mouse weight threshold from comments if available
                if scales_ts.comments:
                    threshold_match = re.search(r'Threshold set to ([\d.]+)g', scales_ts.comments)
                    if threshold_match:
                        self.mouse_weight_threshold = float(threshold_match.group(1))
                    else:
                        self.mouse_weight_threshold = 20.0  # Default
                else:
                    self.mouse_weight_threshold = 20.0
                    
                print(f"Loaded scales data: {len(self.scales_data['timestamps'])} readings")
                print(f"Mouse weight threshold: {self.mouse_weight_threshold}g")
    
    def analyze_scales_platform_events(self, min_duration=1.0):
        """
        Find all instances where the mouse was on the platform (above threshold) for at least min_duration seconds.
        
        Args:
            min_duration (float): Minimum time in seconds the mouse must be on platform
            
        Returns:
            list: List of dictionaries containing platform event info
        """
        if self.scales_data is None:
            print("No scales data available for platform analysis")
            return []
        
        weights = np.array(self.scales_data['data'])
        timestamps = np.array(self.scales_data['timestamps'])
        
        # Find where weight is above threshold
        above_threshold = weights >= self.mouse_weight_threshold
        
        # Find transitions (on and off platform)
        # Add False at start and end to catch edge cases
        padded = np.concatenate(([False], above_threshold, [False]))
        diff = np.diff(padded.astype(int))
        
        # Get start and end indices
        on_platform_starts = np.where(diff == 1)[0]
        on_platform_ends = np.where(diff == -1)[0]
        
        # Calculate durations and filter by minimum duration
        platform_events = []
        for start_idx, end_idx in zip(on_platform_starts, on_platform_ends):
            if start_idx < len(timestamps) and end_idx <= len(timestamps):
                start_time = timestamps[start_idx]
                end_time = timestamps[end_idx - 1] if end_idx > 0 else timestamps[-1]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    platform_events.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
        
        return platform_events
    
    def get_all_pc_trials(self):
        """
        Get ALL trials from PC logs across all ports, not just port 4.
        """
        pc_logs = self.load_pc_logs()
        if not pc_logs:
            return []
        
        all_trials = []
        i = 0
        
        while i < len(pc_logs):
            log = pc_logs[i]
            
            # Look for trial start (OUT message)
            if log['direction'] == 'OUT':
                trial_info = {
                    'out_time': log['timestamp'],
                    'port': None,
                    'response_time': None,
                    'outcome': None,
                    'outcome_time': None,
                    'port_touched': None,
                    'complete': False,
                    'estimated_cue_start': None
                }
                
                # Look for response message (IN;time;R;port;R)
                if i + 1 < len(pc_logs) and pc_logs[i + 1]['direction'] == 'IN':
                    response_log = pc_logs[i + 1]
                    if len(response_log['details']) >= 3 and response_log['details'][0] == 'R':
                        trial_info['response_time'] = response_log['timestamp']
                        try:
                            port = int(response_log['details'][1])
                            trial_info['port'] = port
                        except ValueError:
                            trial_info['port'] = response_log['details'][1]
                
                # Skip the "OUT" message for 's' signal if present
                next_idx = i + 2
                if (next_idx < len(pc_logs) and 
                    pc_logs[next_idx]['direction'] == 'OUT' and 
                    trial_info['response_time'] and
                    pc_logs[next_idx]['timestamp'] - trial_info['response_time'] < 0.1):
                    next_idx += 1
                
                # Look for outcome message (IN;time;C;port;T/F)
                if next_idx < len(pc_logs) and pc_logs[next_idx]['direction'] == 'IN':
                    outcome_log = pc_logs[next_idx]
                    if len(outcome_log['details']) >= 3 and outcome_log['details'][0] == 'C':
                        trial_info['outcome_time'] = outcome_log['timestamp']
                        
                        # Get port touched
                        port_str = outcome_log['details'][1]
                        if port_str != 'F':
                            try:
                                trial_info['port_touched'] = int(port_str)
                            except ValueError:
                                trial_info['port_touched'] = None
                        else:
                            trial_info['port_touched'] = None
                        
                        # Determine outcome
                        outcome_flag = outcome_log['details'][2] if len(outcome_log['details']) > 2 else 'F'
                        if outcome_flag == 'T':
                            trial_info['outcome'] = 'success'
                        elif outcome_flag == 'F':
                            if port_str == 'F':
                                trial_info['outcome'] = 'timeout'
                            else:
                                trial_info['outcome'] = 'failure'
                        
                        trial_info['complete'] = True
                
                # Add ALL trials with valid port and response time
                if trial_info['port'] is not None and trial_info['response_time']:
                    # Estimate LED onset time
                    if trial_info['outcome'] == 'success':
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - 0.5
                    else:
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - 0.1
                    
                    all_trials.append(trial_info)
                
                # Move to next trial
                if trial_info['complete']:
                    i = next_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        return all_trials
    
    def analyze_trial_counts(self):
        """
        Analyse and compare trial counts from different sources.
        Returns a dictionary with trial count information.
        """
        # Get platform events
        platform_events = self.analyze_scales_platform_events(min_duration=1.0)
        
        # Get all PC trials
        self.all_pc_trials = self.get_all_pc_trials()
        
        # Count PC trials by port
        port_counts = {}
        for trial in self.all_pc_trials:
            port = trial.get('port', 'unknown')
            port_counts[port] = port_counts.get(port, 0) + 1
        
        # Create summary
        analysis = {
            'platform_events': len(platform_events),
            'total_pc_trials': len(self.all_pc_trials),
            'led4_trials': len(self.trials),
            'pc_trials_by_port': port_counts,
            'platform_events_list': platform_events[:5] if platform_events else []  # First 5 for display
        }
        
        return analysis
    
    def print_trial_count_comparison(self):
        """
        Print a clear comparison of trial counts from different sources.
        """
        analysis = self.analyze_trial_counts()
        
        print("\n" + "="*60)
        print("TRIAL COUNT COMPARISON")
        print("="*60)
        
        print(f"\nPlatform Events (mouse on scales ≥1s): {analysis['platform_events']}")
        print(f"Total PC Log Trials (all ports): {analysis['total_pc_trials']}")
        print(f"LED_4 Trials Detected: {analysis['led4_trials']}")
        
        print(f"\nDifference (platform events - total trials): {analysis['platform_events'] - analysis['total_pc_trials']}")
        
        # Show breakdown by port
        if analysis['pc_trials_by_port']:
            print(f"\nPC Log Trials by Port:")
            for port in sorted(analysis['pc_trials_by_port'].keys()):
                count = analysis['pc_trials_by_port'][port]
                print(f"  Port {port}: {count} trials")
        
        # Show first few platform events
        if analysis['platform_events_list']:
            print(f"\nFirst {len(analysis['platform_events_list'])} Platform Events:")
            print("-" * 60)
            print(f"{'Event':<6} {'Start Time':<12} {'End Time':<12} {'Duration':<10}")
            print("-" * 60)
            for i, event in enumerate(analysis['platform_events_list']):
                print(f"{i:<6} {event['start_time']:<12.3f} {event['end_time']:<12.3f} {event['duration']:<10.3f}s")
            
            if analysis['platform_events'] > len(analysis['platform_events_list']):
                print(f"... and {analysis['platform_events'] - len(analysis['platform_events_list'])} more events")
    
    def detect_trials(self, compare_methods=True, analyze_scales=True):
        """
        Main method to detect LED_4 trials.
        
        Args:
            compare_methods (bool): Whether to also detect trials from PC logs and compare
            analyze_scales (bool): Whether to analyze scales data and compare trial counts
        """
        print(f"\nDetecting LED_4 trials for session {self.session_id}")
        print("=" * 60)
        
        # Load data
        self.load_daq_data()
        
        # Detect LED trials
        self.trials = self.detect_led4_trials()
        
        # Add sensor and valve data
        self.add_sensor_data_to_trials(self.trials)
        self.add_valve_data_to_trials(self.trials)
        
        # Print detailed results
        self.print_trial_details()
        
        # Analyze scales and compare trial counts
        if analyze_scales:
            self.print_trial_count_comparison()
        
        # Compare with PC log detection if requested
        if compare_methods:
            self.detect_pc_log_trials_port4()
            self.compare_detection_methods()
        
        return self.trials
    
    def detect_led4_trials(self):
        """
        Detect trials from LED_4 on/off events.
        """
        data = self.led4_data['data']
        timestamps = self.led4_data['timestamps']
        
        # Check if LED is stuck high
        unique_values = np.unique(data)
        if len(unique_values) == 1 and unique_values[0] == 1:
            print("WARNING: LED_4 appears to be stuck high")
            return []
        
        # Find transitions
        transitions = np.diff(data)
        num_transitions = np.sum(transitions != 0)
        print(f"LED_4 transitions found: {num_transitions}")
        
        # Determine starting point (should start low)
        if data[0] == 1:
            start_idx = 0  # First high is a trial start
        else:
            start_idx = 1  # First transition to high is trial start
        
        # Extract trial start/end times
        trials = []
        for i in range(start_idx, len(data), 2):
            if i < len(data):
                trial = {
                    'trial_no': len(trials),
                    'correct_port': '4',
                    'cue_start': timestamps[i],
                    'cue_end': timestamps[i + 1] if i + 1 < len(timestamps) else None,
                    'phase': self.phase
                }
                trials.append(trial)
        
        print(f"Detected {len(trials)} LED_4 trials")
        return trials
    
    def add_sensor_data_to_trials(self, trials, trial_timeout=10.0):
        """
        Find sensor touches for each trial.
        
        Args:
            trials: List of trial dictionaries
            trial_timeout: Maximum time (seconds) after cue start to count as valid response
        """
        for j, trial in enumerate(trials):
            trial['sensor_touches'] = []
            
            # Define time window for this trial
            start_time = trial['cue_start']
            timeout_time = trial['cue_start'] + trial_timeout  # 10 second timeout
            
            # Use the earlier of: timeout, next trial start, or session end
            if j + 1 < len(trials):
                end_time = min(timeout_time, trials[j + 1]['cue_start'])
            else:
                end_time = min(timeout_time, self.last_timestamp)
            
            # Check all sensors
            for sensor_num, sensor_info in self.sensor_data.items():
                sensor_timestamps = sensor_info['timestamps']
                sensor_values = sensor_info['data']
                
                # Find sensor activations in time window
                start_idx = bisect.bisect_left(sensor_timestamps, start_time)
                end_idx = bisect.bisect_left(sensor_timestamps, end_time)
                
                for idx in range(start_idx, end_idx):
                    if sensor_values[idx] == 1:
                        touch = {
                            'sensor_touched': str(sensor_num),
                            'sensor_start': sensor_timestamps[idx],
                            'sensor_end': sensor_timestamps[idx + 1] if idx + 1 < len(sensor_timestamps) else None
                        }
                        trial['sensor_touches'].append(touch)
            
            # Sort touches by time
            trial['sensor_touches'].sort(key=lambda x: x['sensor_start'])
            
            # Find first touch after cue AND within timeout window
            trial['next_sensor'] = {}
            for touch in trial['sensor_touches']:
                if (touch['sensor_start'] > trial['cue_start'] and 
                    touch['sensor_start'] <= timeout_time):
                    trial['next_sensor'] = touch
                    break
            
            # Determine success
            if trial['next_sensor']:
                trial['success'] = trial['next_sensor']['sensor_touched'] == '4'
                trial['outcome'] = 'success' if trial['success'] else 'failure'
                trial['response_time'] = trial['next_sensor']['sensor_start']
            else:
                trial['success'] = False
                trial['outcome'] = 'timeout'
                trial['response_time'] = None
    
    def add_valve_data_to_trials(self, trials):
        """
        Match valve actuations to successful trials.
        """
        if 4 not in self.valve_data:
            print("No valve data for port 4")
            return
        
        valve_times = self.valve_data[4]['timestamps']
        valve_idx = 0
        
        for trial in trials:
            trial['valve_time'] = None
            
            if trial['success'] and valve_idx < len(valve_times):
                # Look for valve actuation after response
                if trial['response_time']:
                    # Find valve within 2 seconds of response
                    for idx in range(valve_idx, len(valve_times)):
                        if trial['response_time'] <= valve_times[idx] <= trial['response_time'] + 2.0:
                            trial['valve_time'] = valve_times[idx]
                            valve_idx = idx + 1
                            break
    
    def load_pc_logs(self):
        """
        Load and parse PC logs from behaviour data JSON file.
        """
        if not self.behaviour_data_path.exists():
            print(f"Warning: Behaviour data file not found: {self.behaviour_data_path}")
            return []
        
        with open(self.behaviour_data_path, 'r') as f:
            behaviour_data = json.load(f)
        
        # Extract and clean logs
        raw_logs = behaviour_data.get("Logs", [])
        pc_logs = []
        
        for log in raw_logs:
            # Remove ANSI colour codes
            clean_log = re.sub(r'\u001b\[\d+m', '', log)
            # Parse log entry
            parts = clean_log.split(';')
            if len(parts) >= 2:
                pc_logs.append({
                    'direction': parts[0],
                    'timestamp': float(parts[1]),
                    'details': parts[2:] if len(parts) > 2 else []
                })
        
        print(f"Loaded {len(pc_logs)} PC log entries")
        return pc_logs
    
    def detect_pc_log_trials_port4(self, valve_open_time=0.5):
        """
        Detect port 4 trials from PC logs and estimate LED onset times.
        
        Args:
            valve_open_time (float): Valve opening duration in seconds (default 0.5s)
        """
        pc_logs = self.load_pc_logs()
        if not pc_logs:
            return []
        
        pc_trials = []
        i = 0
        
        while i < len(pc_logs):
            log = pc_logs[i]
            
            # Look for trial start (OUT message)
            if log['direction'] == 'OUT':
                trial_info = {
                    'out_time': log['timestamp'],
                    'port': None,
                    'response_time': None,
                    'outcome': None,
                    'outcome_time': None,
                    'port_touched': None,
                    'complete': False,
                    'estimated_cue_start': None
                }
                
                # Look for response message (IN;time;R;port;R)
                if i + 1 < len(pc_logs) and pc_logs[i + 1]['direction'] == 'IN':
                    response_log = pc_logs[i + 1]
                    if len(response_log['details']) >= 3 and response_log['details'][0] == 'R':
                        trial_info['response_time'] = response_log['timestamp']
                        try:
                            port = int(response_log['details'][1])
                            trial_info['port'] = port
                        except ValueError:
                            pass
                
                # Skip the "OUT" message for 's' signal if present
                next_idx = i + 2
                if (next_idx < len(pc_logs) and 
                    pc_logs[next_idx]['direction'] == 'OUT' and 
                    pc_logs[next_idx]['timestamp'] - trial_info['response_time'] < 0.1):
                    next_idx += 1
                
                # Look for outcome message (IN;time;C;port;T/F)
                if next_idx < len(pc_logs) and pc_logs[next_idx]['direction'] == 'IN':
                    outcome_log = pc_logs[next_idx]
                    if len(outcome_log['details']) >= 3 and outcome_log['details'][0] == 'C':
                        trial_info['outcome_time'] = outcome_log['timestamp']
                        
                        # Get port touched
                        port_str = outcome_log['details'][1]
                        if port_str != 'F':
                            try:
                                trial_info['port_touched'] = int(port_str)
                            except ValueError:
                                trial_info['port_touched'] = None
                        else:
                            trial_info['port_touched'] = None
                        
                        # Determine outcome
                        outcome_flag = outcome_log['details'][2] if len(outcome_log['details']) > 2 else 'F'
                        if outcome_flag == 'T':
                            trial_info['outcome'] = 'success'
                        elif outcome_flag == 'F':
                            if port_str == 'F':
                                trial_info['outcome'] = 'timeout'
                            else:
                                trial_info['outcome'] = 'failure'
                        
                        trial_info['complete'] = True
                
                # Only keep port 4 trials
                if trial_info['port'] == 4 and trial_info['response_time']:
                    # Estimate LED onset time based on outcome
                    if trial_info['outcome'] == 'success':
                        # For successful trials: LED onset = response time - valve duration
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - valve_open_time
                    elif trial_info['outcome'] == 'failure':
                        # For failed trials: LED onset ≈ response time - small processing delay
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - 0.1
                    elif trial_info['outcome'] == 'timeout':
                        # For timeouts: LED onset ≈ response time - small processing delay
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - 0.1
                    else:
                        # No outcome recorded, estimate from response
                        trial_info['estimated_cue_start'] = trial_info['response_time'] - 0.1
                    
                    pc_trials.append(trial_info)
                
                # Move to next trial
                if trial_info['complete']:
                    i = next_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        print(f"\nPC Log Analysis for Port 4:")
        print(f"Found {len(pc_trials)} port 4 trials in PC logs")
        
        # Count outcomes
        successes = sum(1 for t in pc_trials if t['outcome'] == 'success')
        failures = sum(1 for t in pc_trials if t['outcome'] == 'failure')
        timeouts = sum(1 for t in pc_trials if t['outcome'] == 'timeout')
        incomplete = sum(1 for t in pc_trials if not t['outcome'])
        
        print(f"  Successful: {successes}")
        print(f"  Failed: {failures}")
        print(f"  Timeouts: {timeouts}")
        print(f"  Incomplete: {incomplete}")
        
        # Debug: Show first few trials
        if pc_trials:
            print("\nFirst few PC log trials:")
            for i, trial in enumerate(pc_trials[:5]):
                print(f"  Trial {i}: Port {trial['port']}, Outcome: {trial['outcome']}, "
                      f"Complete: {trial['complete']}, Est cue: {trial['estimated_cue_start']:.3f}")
        
        self.pc_log_trials = pc_trials
        return pc_trials
    
    def print_trial_details(self):
        """
        Print detailed information about each trial.
        """
        print(f"\nTrial Details for LED_4")
        print("=" * 60)
        
        # Summary statistics
        total_trials = len(self.trials)
        successful = sum(1 for t in self.trials if t['success'])
        failed = sum(1 for t in self.trials if t['outcome'] == 'failure')
        timeouts = sum(1 for t in self.trials if t['outcome'] == 'timeout')
        valve_matched = sum(1 for t in self.trials if t.get('valve_time'))
        
        print(f"Total trials: {total_trials}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Timeouts: {timeouts}")
        print(f"Valve matched: {valve_matched}")
        print()
        
        # Detailed trial information
        print("Individual trials:")
        print("-" * 80)
        print(f"{'Trial':<6} {'Cue Start':<12} {'Cue End':<12} {'Response':<12} {'Port':<6} {'Outcome':<10} {'Valve':<12}")
        print("-" * 80)
        
        for trial in self.trials[:20]:  # Show first 20 trials
            cue_start = f"{trial['cue_start']:.3f}"
            cue_end = f"{trial['cue_end']:.3f}" if trial['cue_end'] else "None"
            response = f"{trial['response_time']:.3f}" if trial.get('response_time') else "None"
            port_touched = trial['next_sensor'].get('sensor_touched', '-') if trial['next_sensor'] else '-'
            outcome = trial['outcome']
            valve = f"{trial['valve_time']:.3f}" if trial.get('valve_time') else "None"
            
            print(f"{trial['trial_no']:<6} {cue_start:<12} {cue_end:<12} {response:<12} {port_touched:<6} {outcome:<10} {valve:<12}")
        
        if len(self.trials) > 20:
            print(f"... and {len(self.trials) - 20} more trials")
        
        print("-" * 80)
        
        # Timing analysis
        if successful > 0:
            response_times = [t['response_time'] - t['cue_start'] 
                            for t in self.trials 
                            if t['success'] and t.get('response_time')]
            if response_times:
                print(f"\nResponse time statistics (successful trials):")
                print(f"  Mean: {np.mean(response_times):.3f}s")
                print(f"  Median: {np.median(response_times):.3f}s")
                print(f"  Min: {np.min(response_times):.3f}s")
                print(f"  Max: {np.max(response_times):.3f}s")
    
    def compare_detection_methods(self):
        """
        Compare trials detected from DAQ traces vs PC logs by matching successful trials in order.
        """
        print("\n" + "="*60)
        print("COMPARISON: DAQ Detection vs PC Log Detection")
        print("="*60)
        
        # Get PC log trials if not already loaded
        if not self.pc_log_trials:
            self.detect_pc_log_trials_port4()
        
        # Summary counts
        print(f"\nTotal trials detected:")
        print(f"  DAQ method: {len(self.trials)}")
        print(f"  PC log method: {len(self.pc_log_trials)}")
        
        # Get successful trials from each method
        daq_successes = [(i, t) for i, t in enumerate(self.trials) if t['success']]
        pc_successes = [(i, t) for i, t in enumerate(self.pc_log_trials) if t['outcome'] == 'success']
        
        print(f"\nSuccessful trials:")
        print(f"  DAQ method: {len(daq_successes)}")
        print(f"  PC log method: {len(pc_successes)}")
        
        # Match successful trials in order
        matched_count = min(len(daq_successes), len(pc_successes))
        
        if matched_count > 0:
            print(f"\nMatching {matched_count} successful trials in order")
            print("-" * 100)
            print(f"{'Match':<6} {'DAQ#':<6} {'PC#':<6} {'Valve Time':<12} {'PC Outcome':<12} {'Est. Msg Time':<15} {'LED Est.':<12} {'LED Actual':<12} {'Diff':<8}")
            print("-" * 100)
            
            led_timing_errors = []
            
            for match_idx in range(matched_count):
                daq_idx, daq_trial = daq_successes[match_idx]
                pc_idx, pc_trial = pc_successes[match_idx]
                
                # Get valve time from DAQ data
                valve_time = daq_trial.get('valve_time')
                
                if valve_time:
                    # For successful trials, the Arduino sends the message after valve closes
                    # Valve duration is 0.5s, so message sent at valve_time + 0.5s
                    valve_duration = 0.5  # 500ms
                    arduino_processing_delay = 0.015  # ~15ms for Arduino to process and send message
                    estimated_msg_time = valve_time + valve_duration + arduino_processing_delay
                    
                    # Time between OUT message and outcome message in PC logs
                    pc_time_diff = pc_trial['outcome_time'] - pc_trial['out_time']
                    
                    # LED turned on at: message_sent_time - pc_time_diff
                    new_led_estimate = estimated_msg_time - pc_time_diff
                    
                    # Compare to actual LED time from DAQ
                    actual_led_time = daq_trial['cue_start']
                    timing_error = new_led_estimate - actual_led_time
                    led_timing_errors.append(timing_error)
                    
                    print(f"{match_idx:<6} {daq_idx:<6} {pc_idx:<6} {valve_time:<12.3f} {pc_trial['outcome_time']:<12.3f} "
                          f"{estimated_msg_time:<15.3f} {new_led_estimate:<12.3f} {actual_led_time:<12.3f} {timing_error:<8.3f}")
                else:
                    print(f"{match_idx:<6} {daq_idx:<6} {pc_idx:<6} {'No valve':<12} {pc_trial['outcome_time']:<12.3f} "
                          f"{'N/A':<15} {'N/A':<12} {daq_trial['cue_start']:<12.3f} {'N/A':<8}")
            
            if led_timing_errors:
                print("\nLED timing estimation accuracy:")
                print(f"  Mean error: {np.mean(led_timing_errors):.3f}s")
                print(f"  Std deviation: {np.std(led_timing_errors):.3f}s")
                print(f"  Median error: {np.median(led_timing_errors):.3f}s")
                print(f"  Min error: {np.min(led_timing_errors):.3f}s")
                print(f"  Max error: {np.max(led_timing_errors):.3f}s")
                
                # Calculate a correction factor
                correction = -np.median(led_timing_errors)
                print(f"\nSuggested correction factor: {correction:.3f}s")
                print("(Add this to your PC log LED estimates to improve accuracy)")
        
        # Also show comparison for all trial types
        print(f"\n\nOutcome comparison (all trials):")
        print("-" * 50)
        
        # Count outcomes for each method
        daq_outcomes = {'success': 0, 'failure': 0, 'timeout': 0}
        pc_outcomes = {'success': 0, 'failure': 0, 'timeout': 0, 'none': 0}
        
        for trial in self.trials:
            daq_outcomes[trial['outcome']] += 1
            
        for trial in self.pc_log_trials:
            outcome = trial['outcome'] if trial['outcome'] else 'none'
            pc_outcomes[outcome] += 1
        
        print(f"{'Outcome':<15} {'DAQ Count':<12} {'PC Log Count':<12}")
        print("-" * 50)
        for outcome in ['success', 'failure', 'timeout']:
            print(f"{outcome:<15} {daq_outcomes.get(outcome, 0):<12} {pc_outcomes.get(outcome, 0):<12}")
        if pc_outcomes.get('none', 0) > 0:
            print(f"{'incomplete':<15} {0:<12} {pc_outcomes['none']:<12}")
    
    def plot_trial_traces(self, output_dir, buffer_before=0.5, buffer_after=4.0, trials_to_plot=None):
        """
        Plot DAQ traces for each trial showing LED, valve, and sensor signals.
        
        Args:
            output_dir (str or Path): Directory to save plots
            buffer_before (float): Time buffer before trial start (seconds)
            buffer_after (float): Time buffer after trial end (seconds)
            trials_to_plot (list): List of trial numbers to plot (None for all)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if trials_to_plot is None:
            trials_to_plot = range(len(self.trials))
        
        print(f"\nPlotting trial traces to {output_path}")
        
        for trial_idx in trials_to_plot:
            if trial_idx >= len(self.trials):
                continue
                
            trial = self.trials[trial_idx]
            
            # Define time window
            start_time = trial['cue_start'] - buffer_before
            end_time = (trial['cue_end'] if trial['cue_end'] else trial['cue_start'] + 10) + buffer_after
            
            # Create figure
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"Trial {trial['trial_no']} - Port 4 - {trial['outcome'].upper()}", fontsize=14)
            
            # Plot LED trace
            self._plot_trace(axes[0], self.led4_data, start_time, end_time, 
                           'LED_4', 'blue', trial)
            
            # Plot valve trace
            if 4 in self.valve_data:
                self._plot_trace(axes[1], self.valve_data[4], start_time, end_time, 
                               'VALVE4', 'green', trial)
            else:
                axes[1].text(0.5, 0.5, 'No valve data', transform=axes[1].transAxes,
                           ha='center', va='center', fontsize=12, color='gray')
                axes[1].set_ylabel('VALVE4')
            
            # Plot sensor traces (all 6 on one axis)
            self._plot_sensor_traces(axes[2], start_time, end_time, trial)
            
            # Plot scales data
            self._plot_scales_data(axes[3], start_time, end_time, trial)
            
            # Add vertical lines for key events
            for ax in axes:
                # Cue start
                ax.axvline(trial['cue_start'], color='black', linestyle='--', 
                          alpha=0.5, label='Cue start')
                # Cue end
                if trial['cue_end']:
                    ax.axvline(trial['cue_end'], color='black', linestyle=':', 
                              alpha=0.5, label='Cue end')
                # Trial timeout (10 seconds after cue start)
                timeout_time = trial['cue_start'] + 10.0
                ax.axvline(timeout_time, color='orange', linestyle='--', 
                          alpha=0.5, label='Timeout (10s)')
                # Response time
                if trial.get('response_time'):
                    ax.axvline(trial['response_time'], color='red', linestyle='--', 
                              alpha=0.5, label='Response')
                # Valve time
                if trial.get('valve_time'):
                    ax.axvline(trial['valve_time'], color='green', linestyle='--', 
                              alpha=0.5, label='Valve')
            
            # Format x-axis
            axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            axes[-1].set_xlim(start_time, end_time)
            
            # Add legend to top plot
            axes[0].legend(loc='upper right', fontsize=9)
            
            # Add trial info text
            info_text = f"Cue: {trial['cue_start']:.3f}s"
            if trial.get('response_time'):
                response_delay = trial['response_time'] - trial['cue_start']
                info_text += f"\nResponse: {trial['response_time']:.3f}s (+{response_delay:.3f}s)"
            if trial['next_sensor']:
                info_text += f"\nPort touched: {trial['next_sensor']['sensor_touched']}"
            
            axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Save figure
            outcome_str = trial['outcome']
            port_touched = trial['next_sensor'].get('sensor_touched', 'none') if trial['next_sensor'] else 'none'
            filename = f"trial_{trial['trial_no']:03d}_{outcome_str}_touched{port_touched}.png"
            
            plt.tight_layout()
            fig.savefig(output_path / filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {filename}")
        
        print(f"Completed plotting {len(trials_to_plot)} trials")
    
    def _plot_trace(self, ax, data_dict, start_time, end_time, label, color, trial):
        """
        Plot a single trace (LED or valve) within the time window.
        """
        timestamps = data_dict['timestamps']
        values = data_dict['data']
        
        # Find indices within time window
        start_idx = bisect.bisect_left(timestamps, start_time)
        end_idx = bisect.bisect_right(timestamps, end_time)
        
        if start_idx < end_idx:
            # For valve data, we need to create a proper trace
            if 'VALVE' in label:
                # Valve data is stored as timestamps and durations
                # Create a trace showing valve open/closed states
                plot_times = []
                plot_values = []
                
                for i in range(start_idx, end_idx):
                    valve_start = timestamps[i]
                    valve_duration = values[i]  # Duration in seconds
                    valve_end = valve_start + valve_duration
                    
                    # Add points for valve opening and closing
                    plot_times.extend([valve_start, valve_start, valve_end, valve_end])
                    plot_values.extend([0, 1, 1, 0])
                
                if plot_times:
                    ax.plot(plot_times, plot_values, color=color, linewidth=2, label=label)
                    
                    # Fill under the valve pulses
                    for i in range(start_idx, end_idx):
                        valve_start = timestamps[i]
                        valve_duration = values[i]
                        valve_end = valve_start + valve_duration
                        ax.fill_between([valve_start, valve_end], 0, 1, alpha=0.3, color=color)
            else:
                # LED data - plot as before
                t = timestamps[start_idx:end_idx]
                v = values[start_idx:end_idx]
                
                # Plot as step function
                ax.step(t, v, where='post', color=color, linewidth=2, label=label)
                ax.fill_between(t, 0, v, step='post', alpha=0.3, color=color)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 1])
    
    def _plot_sensor_traces(self, ax, start_time, end_time, trial):
        """
        Plot all sensor traces on a single axis with different y-offsets.
        """
        colors = plt.cm.tab10(range(6))
        
        for i, (sensor_num, sensor_data) in enumerate(self.sensor_data.items()):
            timestamps = sensor_data['timestamps']
            values = sensor_data['data']
            
            # Find indices within time window
            start_idx = bisect.bisect_left(timestamps, start_time)
            end_idx = bisect.bisect_right(timestamps, end_time)
            
            if start_idx < end_idx:
                t = timestamps[start_idx:end_idx]
                v = values[start_idx:end_idx]
                
                # Offset each sensor trace
                y_offset = sensor_num - 1
                v_offset = v * 0.8 + y_offset  # Scale to 0.8 height
                
                # Plot
                ax.step(t, v_offset, where='post', color=colors[i], 
                       linewidth=2, label=f'Sensor {sensor_num}')
                
                # Highlight if this sensor was touched
                if (trial['next_sensor'] and 
                    trial['next_sensor']['sensor_touched'] == str(sensor_num)):
                    ax.fill_between(t, y_offset, v_offset, step='post', 
                                  alpha=0.5, color=colors[i])
        
        ax.set_ylabel('Sensors', fontsize=11)
        ax.set_ylim(-0.5, 6.5)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f'S{i+1}' for i in range(6)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    
    def _plot_scales_data(self, ax, start_time, end_time, trial):
        """
        Plot scales/weight data within the time window.
        """
        if self.scales_data is None:
            ax.text(0.5, 0.5, 'No scales data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_ylabel('Weight (g)')
            return
        
        timestamps = self.scales_data['timestamps']
        weights = self.scales_data['data']
        
        # Find indices within time window
        start_idx = bisect.bisect_left(timestamps, start_time)
        end_idx = bisect.bisect_right(timestamps, end_time)
        
        if start_idx < end_idx:
            t = timestamps[start_idx:end_idx]
            w = weights[start_idx:end_idx]
            
            # Plot weight data
            ax.plot(t, w, color='purple', linewidth=2, label='Weight')
            
            # Add horizontal line for mouse weight threshold
            if hasattr(self, 'mouse_weight_threshold'):
                ax.axhline(y=self.mouse_weight_threshold, color='red', linestyle='--', 
                          alpha=0.5, label=f'Threshold ({self.mouse_weight_threshold}g)')
            
            # Highlight when mouse is on platform (above threshold)
            if hasattr(self, 'mouse_weight_threshold'):
                # Create a boolean mask for when weight is above threshold
                above_threshold = np.array(w) >= self.mouse_weight_threshold
                
                # Find continuous regions above threshold
                if np.any(above_threshold):
                    # Get start and end indices of continuous regions
                    diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    # Shade regions where mouse is on platform
                    for start, end in zip(starts, ends):
                        if start < len(t) and end <= len(t):
                            ax.axvspan(t[start], t[end-1], alpha=0.2, color='green', 
                                     label='On platform' if start == starts[0] else '')
        
        ax.set_ylabel('Weight (g)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        if start_idx < end_idx and len(w) > 0:
            y_min = min(0, np.min(w) - 5)
            y_max = max(np.max(w) + 5, self.mouse_weight_threshold + 10 if hasattr(self, 'mouse_weight_threshold') else np.max(w) + 5)
            ax.set_ylim(y_min, y_max)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)


def validate_with_led4(session_dict, plot_traces=False, output_dir=None, trials_to_plot=None, compare_methods=True, analyze_scales=True):
    """
    Run LED_4 trial detection for validation.
    
    Args:
        session_dict (dict): Session dictionary from Cohort_folder
        plot_traces (bool): Whether to plot trial traces
        output_dir (str or Path): Directory to save plots (required if plot_traces=True)
        trials_to_plot (list): List of trial numbers to plot (None for all)
        compare_methods (bool): Whether to compare DAQ and PC log detection methods
        analyze_scales (bool): Whether to analyze scales data and compare trial counts
    """
    detector = LED4TrialDetector(session_dict)
    trials = detector.detect_trials(compare_methods=compare_methods, analyze_scales=analyze_scales)
    
    if plot_traces and output_dir:
        detector.plot_trial_traces(output_dir, trials_to_plot=trials_to_plot)
    elif plot_traces and not output_dir:
        print("Warning: plot_traces=True but no output_dir provided. Skipping plots.")
    
    return detector


# Example usage:
if __name__ == "__main__":
    print("Loading cohort info...")
    cohort = Cohort_folder('/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics')
    print("Cohort info loaded.")
    
    session_dict = cohort.get_session("250624_143350_mtao101-3d")
    
    # Detect trials with scales analysis
    detector = validate_with_led4(
        session_dict, 
        plot_traces=False,
        analyze_scales=True,  # This will show the trial count comparison
        compare_methods=True
    )