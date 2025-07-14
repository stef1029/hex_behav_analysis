"""
Streamlined scales-based alignment for integration into trial finding.
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
import json
import re
from scipy.optimize import minimize_scalar
from scipy.stats import linregress


class ScalesTrialRecovery:
    """
    Recover trial times using scales platform events when LED traces are corrupted.
    """
    
    def __init__(self, session_dict, rig_id):
        """
        Initialise the recovery system.
        
        Args:
            session_dict (dict): Session dictionary from Cohort_folder
        """
        self.session_dict = session_dict
        self.session_id = session_dict.get('session_id')
        
        # Check rig - only works on rigs with accurate scales timestamps
        self.rig = rig_id
        if self.rig in ['behaviour-rig1', 'behaviour-rig2']:
            self.scales_reliable = False
        else:
            self.scales_reliable = True
        
        # File paths
        if session_dict.get('portable'):
            self.nwb_file_path = Path(session_dict.get('NWB_file'))
        else:
            self.nwb_file_path = Path(session_dict.get('processed_data', {}).get('NWB_file'))
        
        self.behaviour_data_path = Path(session_dict.get('raw_data', {}).get('behaviour_data'))
        
        # Data containers
        self.scales_data = None
        self.mouse_weight_threshold = 20.0
        self.platform_events = []
        self.pc_trials_all = []  # All PC trials
        self.aligned_trials = None
        self.transformation_params = None
        self.validation_results = None
        
    def is_available(self):
        """Check if scales-based recovery is available for this session."""
        return self.scales_reliable and self.behaviour_data_path.exists()
    
    def run_recovery(self):
        """
        Run the complete recovery process if not already done.
        Returns aligned trials with DAQ timestamps.
        """
        if self.aligned_trials is not None:
            return self.aligned_trials
        
        if not self.is_available():
            raise ValueError(f"Scales-based recovery not available for rig {self.rig}")
        
        # Load all data
        self._load_scales_data()
        self._get_platform_events()
        self._load_all_pc_trials()
        
        # Run three-stage alignment
        self._run_alignment()
        
        return self.aligned_trials
    
    def _load_scales_data(self):
        """Load scales data from NWB file."""
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            nwbfile = io.read()
            
            if "scales" in nwbfile.acquisition:
                scales_ts = nwbfile.acquisition["scales"]
                self.scales_data = {
                    'data': scales_ts.data[:],
                    'timestamps': scales_ts.timestamps[:]
                }
                
                # Extract threshold
                if scales_ts.comments:
                    threshold_match = re.search(r'Threshold set to ([\d.]+)g', scales_ts.comments)
                    if threshold_match:
                        self.mouse_weight_threshold = float(threshold_match.group(1))
            else:
                raise ValueError("No scales data found in NWB file")
    
    def _get_platform_events(self):
        """Extract platform events from scales data."""
        weights = np.array(self.scales_data['data'])
        timestamps = np.array(self.scales_data['timestamps'])
        
        # Find where weight is above threshold
        above_threshold = weights >= self.mouse_weight_threshold
        
        # Find transitions
        padded = np.concatenate(([False], above_threshold, [False]))
        diff = np.diff(padded.astype(int))
        
        on_starts = np.where(diff == 1)[0]
        on_ends = np.where(diff == -1)[0]
        
        # Get events with sufficient duration (≥1 second)
        event_times = []
        for start_idx, end_idx in zip(on_starts, on_ends):
            if start_idx < len(timestamps) and end_idx <= len(timestamps):
                start_time = timestamps[start_idx]
                end_time = timestamps[end_idx - 1] if end_idx > 0 else timestamps[-1]
                duration = end_time - start_time
                
                if duration >= 1.0:
                    # Add 1 second for expected trial start
                    event_times.append(start_time + 1.0)
        
        self.platform_events = event_times
    
    def _load_all_pc_trials(self):
        """Load ALL trials from PC logs (all ports) with proper timing adjustments."""
        with open(self.behaviour_data_path, 'r') as f:
            behaviour_data = json.load(f)
        
        # Parse logs
        raw_logs = behaviour_data.get("Logs", [])
        pc_logs = []
        
        for log in raw_logs:
            clean_log = re.sub(r'\u001b\[\d+m', '', log)
            parts = clean_log.split(';')
            if len(parts) >= 2:
                pc_logs.append({
                    'direction': parts[0],
                    'timestamp': float(parts[1]),
                    'details': parts[2:] if len(parts) > 2 else []
                })
        
        # Extract all trials
        trials = []
        i = 0
        
        while i < len(pc_logs):
            log = pc_logs[i]
            
            if log['direction'] == 'OUT':
                trial_info = {
                    'pc_time': log['timestamp'],  # This is the OUT message time (cue start)
                    'port': None,  # Cued port
                    'response_time': None,
                    'response_port': None,  # Port actually touched (from Arduino)
                    'outcome': None,
                    'outcome_time': None,  # Time of outcome message
                    'complete': False
                }
                
                # Look for response message (IN;time;R;port;R)
                if i + 1 < len(pc_logs) and pc_logs[i + 1]['direction'] == 'IN':
                    response_log = pc_logs[i + 1]
                    if len(response_log['details']) >= 3 and response_log['details'][0] == 'R':
                        trial_info['response_time'] = response_log['timestamp']
                        try:
                            trial_info['port'] = int(response_log['details'][1])  # Cued port
                        except ValueError:
                            pass
                
                # Skip intermediate OUT if present
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
                        
                        # Get port touched (what sensor was actually activated)
                        port_str = outcome_log['details'][1]
                        if port_str != 'F':
                            try:
                                trial_info['response_port'] = int(port_str)
                            except ValueError:
                                trial_info['response_port'] = None
                        else:
                            trial_info['response_port'] = None  # Timeout
                        
                        outcome_flag = outcome_log['details'][2] if len(outcome_log['details']) > 2 else 'F'
                        if outcome_flag == 'T':
                            trial_info['outcome'] = 'success'
                        elif outcome_flag == 'F':
                            if port_str == 'F':
                                trial_info['outcome'] = 'timeout'
                            else:
                                trial_info['outcome'] = 'failure'
                        
                        trial_info['complete'] = True
                
                # Add valid trials
                if trial_info['port'] is not None and trial_info['response_time']:
                    trials.append(trial_info)
                
                # Move to next
                if trial_info['complete']:
                    i = next_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        self.pc_trials_all = trials
    
    def _run_alignment(self):
        """Run three-stage alignment process."""
        pc_trial_times = [t['pc_time'] for t in self.pc_trials_all]
        
        # Stage 1: Initial offset
        offset1 = self._find_initial_offset(pc_trial_times)
        matches1 = self._match_trials(pc_trial_times, self.platform_events, offset1, 2.0)
        
        if len(matches1['matched']) < 10:
            raise ValueError("Too few matched trials for alignment")
        
        # Stage 2: Linear transformation
        matched_pc = [m['pc_time'] for m in matches1['matched']]
        matched_platform = [m['platform_time'] for m in matches1['matched']]
        
        slope2, intercept2, r_value, _, _ = linregress(matched_pc, matched_platform)
        
        # Constrain scale
        if abs(slope2 - 1.0) > 0.01:
            slope2 = 1.0
        
        # Stage 3: Refinement with best matches
        matches2 = self._match_trials_linear(pc_trial_times, self.platform_events, 
                                        slope2, intercept2, 0.5)
        
        # Select best matches
        distances = [m['distance'] for m in matches2['matched']]
        mean_dist = np.mean(distances)
        best_matches = [m for m in matches2['matched'] 
                    if abs(m['distance'] - mean_dist) <= 0.08]
        
        if len(best_matches) >= 5:
            best_pc = [m['pc_time'] for m in best_matches]
            best_platform = [m['platform_time'] for m in best_matches]
            slope3, intercept3, _, _, _ = linregress(best_pc, best_platform)
        else:
            slope3, intercept3 = slope2, intercept2
        
        # Apply final transformation with 2ms correction
        self.transformation_params = {
            'scale': slope3,
            'offset': intercept3,
            'correction': 0.002  # 2ms empirical correction
        }
        
        # Create aligned trials with all info preserved
        self.aligned_trials = []
        for trial in self.pc_trials_all:
            aligned_time = (trial['pc_time'] * slope3 + intercept3 + 0.002)
            
            # Calculate aligned sensor time if there was a response
            aligned_response_time = None
            aligned_outcome_time = None
            if trial['response_time']:
                aligned_response_time = trial['response_time'] * slope3 + intercept3 + 0.002
            if trial.get('outcome_time'):
                aligned_outcome_time = trial['outcome_time'] * slope3 + intercept3 + 0.002
                
            self.aligned_trials.append({
                'port': trial['port'],
                'aligned_time': aligned_time,
                'pc_time': trial['pc_time'],
                'outcome': trial['outcome'],
                'response_time': aligned_response_time,
                'outcome_time': aligned_outcome_time,
                'response_port': trial.get('response_port')  # Port actually touched
            })
    def _find_initial_offset(self, pc_times, max_offset=100):
        """Find initial offset using optimization."""
        def cost(offset):
            total = 0
            platform_array = np.array(self.platform_events)
            for pc_time in pc_times:
                adjusted = pc_time + offset
                distances = np.abs(platform_array - adjusted)
                total += np.min(distances) ** 2
            return total
        
        result = minimize_scalar(cost, bounds=(-max_offset, max_offset), method='bounded')
        return result.x
    
    def _match_trials(self, pc_times, platform_times, offset, scale, max_dist=0.5):
        """Match PC trials to platform events."""
        matched = []
        platform_array = np.array(platform_times)
        used_platforms = set()
        
        for i, pc_time in enumerate(pc_times):
            adjusted = pc_time * scale + offset
            distances = np.abs(platform_array - adjusted)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] <= max_dist and nearest_idx not in used_platforms:
                matched.append({
                    'pc_time': pc_time,
                    'platform_time': platform_array[nearest_idx],
                    'distance': distances[nearest_idx]
                })
                used_platforms.add(nearest_idx)
        
        return {'matched': matched}
    
    def _match_trials_linear(self, pc_times, platform_times, scale, offset, max_dist=0.5):
        """Match trials with linear transformation."""
        return self._match_trials(pc_times, platform_times, offset, scale, max_dist)
    
    def get_trials_for_port(self, port):
        """
        Get aligned trial times for a specific port with proper sensor timing.
        
        Returns:
            list: Trial dictionaries with 'cue_start' in DAQ time and sensor info
        """
        if self.aligned_trials is None:
            self.run_recovery()
        
        port_trials = []
        trial_no = 0
        
        for trial in self.aligned_trials:
            if trial['port'] == port:
                # Build sensor info if there was a response
                sensor_touches = []
                next_sensor = {}
                
                if trial['response_port'] is not None and trial['outcome_time']:
                    # Calculate actual sensor touch time based on outcome
                    sensor_time = trial['outcome_time']  # Start with outcome message time
                    
                    if trial['outcome'] == 'success':
                        # For success, subtract valve opening time (500ms)
                        sensor_time = sensor_time - 0.5
                    elif trial['outcome'] == 'failure':
                        # For failure, subtract error signal time (5s)
                        sensor_time = sensor_time - 5.0
                    # For timeout, no adjustment needed
                    
                    sensor_info = {
                        'sensor_touched': str(trial['response_port']),
                        'sensor_start': sensor_time,
                        'sensor_end': sensor_time + 0.1  # Approximate
                    }
                    sensor_touches = [sensor_info]
                    next_sensor = sensor_info
                
                port_trial = {
                    'trial_no': trial_no,
                    'cue_start': trial['aligned_time'],  # Aligned OUT message time
                    'cue_end': None,  # Will be filled by existing code
                    'pc_time': trial['pc_time'],
                    'outcome': trial['outcome'],
                    'correct_port': str(port),
                    'recovered': True,
                    'sensor_touches': sensor_touches,
                    'next_sensor': next_sensor,
                    'success': trial['outcome'] == 'success'
                }
                
                port_trials.append(port_trial)
                trial_no += 1
        
        return port_trials
    
    def validate_against_port4(self, led4_trials):
        """
        Validate recovery accuracy using Port 4 LED trials.
        
        Args:
            led4_trials (list): Trials detected from LED_4 DAQ trace
            
        Returns:
            dict: Validation statistics
        """
        port4_recovered = self.get_trials_for_port(4)
        
        # Match trials
        n_matches = min(len(led4_trials), len(port4_recovered))
        
        if n_matches == 0:
            return None
        
        differences = []
        for i in range(n_matches):
            led_time = led4_trials[i]['cue_start']
            recovered_time = port4_recovered[i]['cue_start']
            differences.append(recovered_time - led_time)
        
        differences = np.array(differences)
        differences_ms = differences * 1000
        
        # Calculate stats
        validation = {
            'n_compared': n_matches,
            'mean_error_ms': np.mean(differences_ms),
            'std_error_ms': np.std(differences_ms),
            'median_error_ms': np.median(differences_ms),
            'rmse_ms': np.sqrt(np.mean(differences_ms**2)),
            'within_1ms': np.sum(np.abs(differences_ms) <= 1.0),
            'within_1ms_percent': np.sum(np.abs(differences_ms) <= 1.0) / n_matches * 100
        }
        
        self.validation_results = validation
        return validation