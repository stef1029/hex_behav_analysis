"""
Script to align scales activation times with PC log trial times.
Attempts to find the time offset between DAQ and PC clocks.
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
import json
import re
from scipy.optimize import minimize_scalar
from scipy.signal import correlate
from scipy.stats import linregress


class ScalesTrialAligner:
    """
    Align scales platform events with PC log trials to identify actual trial starts.
    """
    
    def __init__(self, session_dict):
        """
        Initialise the aligner with session information.
        
        Args:
            session_dict (dict): Session dictionary from Cohort_folder
        """
        self.session_dict = session_dict
        self.session_id = session_dict.get('session_id')
        
        # Load file paths
        if session_dict.get('portable'):
            self.nwb_file_path = Path(session_dict.get('NWB_file'))
        else:
            self.nwb_file_path = Path(session_dict.get('processed_data', {}).get('NWB_file'))
        
        self.behaviour_data_path = Path(session_dict.get('raw_data', {}).get('behaviour_data'))
        
        # Data containers
        self.scales_data = None
        self.mouse_weight_threshold = 20.0  # Default
        self.platform_events = []
        self.pc_trials = []

        # Load mouse weight threshold from behaviour data if available
        if self.behaviour_data_path.exists():
            try:
                with open(self.behaviour_data_path, 'r') as f:
                    behaviour_data = json.load(f)
                    # Get the threshold from the behaviour data
                    if "Mouse weight threshold" in behaviour_data:
                        self.mouse_weight_threshold = float(behaviour_data["Mouse weight threshold"])
            except Exception:
                pass  # Keep default if loading fails
        
    def load_scales_data(self):
        """
        Load scales data from NWB file.
        """
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
                
                print(f"Loaded {len(self.scales_data['timestamps'])} scales readings")
                print(f"Mouse weight threshold: {self.mouse_weight_threshold}g")
            else:
                raise ValueError("No scales data found in NWB file")
    
    def get_platform_events(self, min_duration=1.0):
        """
        Extract platform events (mouse on scales) with duration >= min_duration.
        
        Args:
            min_duration (float): Minimum duration in seconds
            
        Returns:
            list: Platform event times (start_time + 1s for each event)
        """
        weights = np.array(self.scales_data['data'])
        timestamps = np.array(self.scales_data['timestamps'])
        
        # Find where weight is above threshold
        above_threshold = weights >= self.mouse_weight_threshold
        
        # Find transitions
        padded = np.concatenate(([False], above_threshold, [False]))
        diff = np.diff(padded.astype(int))
        
        on_starts = np.where(diff == 1)[0]
        on_ends = np.where(diff == -1)[0]
        
        # Get events with sufficient duration
        event_times = []
        self.platform_event_details = []  # Store details for debugging
        
        for start_idx, end_idx in zip(on_starts, on_ends):
            if start_idx < len(timestamps) and end_idx <= len(timestamps):
                start_time = timestamps[start_idx]
                end_time = timestamps[end_idx - 1] if end_idx > 0 else timestamps[-1]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    # Use start_time + 1s as the expected trial start time
                    event_times.append(start_time + 1.0)
                    self.platform_event_details.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'event_time': start_time + 1.0
                    })
        
        self.platform_events = event_times
        print(f"  Platform events detected: {len(event_times)}")
        print(f"  First few events: {event_times[:5] if event_times else 'None'}")
        
        return event_times
    
    def load_pc_trials(self):
        """
        Load all trials from PC logs.
        Match the logic from the original script to ensure consistency.
        """
        if not self.behaviour_data_path.exists():
            raise FileNotFoundError(f"Behaviour data file not found: {self.behaviour_data_path}")
        
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
        
        # Extract trials using same logic as original script
        trials = []
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
                    'complete': False
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
                
                # Only add trials with valid port and response time
                if trial_info['port'] is not None and trial_info['response_time']:
                    trials.append({
                        'time': trial_info['out_time'],  # Use OUT time as trial start
                        'port': trial_info['port'],
                        'outcome': trial_info['outcome'],
                        'complete': trial_info['complete']
                    })
                
                # Move to next trial
                if trial_info['complete']:
                    i = next_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        # Store trial times and full trial info
        self.pc_trials = [t['time'] for t in trials]
        self.pc_trials_full = trials  # Store full trial information
        
        # Debug info
        print(f"  Total log entries: {len(pc_logs)}")
        print(f"  OUT messages: {sum(1 for log in pc_logs if log['direction'] == 'OUT')}")
        print(f"  Complete trials: {sum(1 for t in trials if t['complete'])}")
        
        return trials
    
    def find_time_offset_minimize(self, max_offset=100):
        """
        Find time offset by minimising the sum of distances from PC trials to nearest platform events.
        
        Args:
            max_offset (float): Maximum offset to search (seconds)
            
        Returns:
            float: Best offset value to add to PC trial times
        """
        if not self.platform_events or not self.pc_trials:
            raise ValueError("No data to align")
        
        def cost_function(offset):
            """
            Cost function: sum of distances from each PC trial to nearest platform event.
            """
            total_cost = 0
            platform_array = np.array(self.platform_events)
            
            for pc_trial_time in self.pc_trials:
                # Adjust PC trial time by offset
                adjusted_time = pc_trial_time + offset
                
                # Find distance to nearest platform event
                distances = np.abs(platform_array - adjusted_time)
                min_distance = np.min(distances)
                
                # Use squared distance to penalise large deviations
                total_cost += min_distance ** 2
            
            return total_cost
        
        # Find optimal offset
        result = minimize_scalar(cost_function, bounds=(-max_offset, max_offset), method='bounded')
        
        return result.x
    
    def align_and_match(self, offset, max_match_distance=0.5):
        """
        Apply offset to PC trial times and match them to platform events.
        
        Args:
            offset (float): Time offset to apply to PC trial times
            max_match_distance (float): Maximum time difference for matching (seconds)
            
        Returns:
            dict: Matching results
        """
        matched_pc_trials = []
        unmatched_pc_trials = []
        matched_platform_events = []
        unmatched_platform_events = []
        
        platform_array = np.array(self.platform_events)
        used_platform_events = set()
        
        # Match each PC trial to nearest platform event
        for i, pc_trial_time in enumerate(self.pc_trials):
            # Adjust PC trial time
            adjusted_time = pc_trial_time + offset
            
            # Find nearest platform event
            distances = np.abs(platform_array - adjusted_time)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance <= max_match_distance and nearest_idx not in used_platform_events:
                # Match found
                matched_pc_trials.append({
                    'pc_trial_time': pc_trial_time,
                    'adjusted_time': adjusted_time,
                    'platform_time': platform_array[nearest_idx],
                    'distance': nearest_distance,
                    'trial_idx': i
                })
                used_platform_events.add(nearest_idx)
            else:
                # No match
                unmatched_pc_trials.append({
                    'pc_trial_time': pc_trial_time,
                    'adjusted_time': adjusted_time,
                    'nearest_distance': nearest_distance,
                    'trial_idx': i
                })
        
        # Find unmatched platform events
        for i, platform_time in enumerate(self.platform_events):
            if i not in used_platform_events:
                unmatched_platform_events.append({
                    'platform_time': platform_time,
                    'platform_idx': i
                })
            else:
                matched_platform_events.append({
                    'platform_time': platform_time,
                    'platform_idx': i
                })
        
        return {
            'offset': offset,
            'matched_pc_trials': matched_pc_trials,
            'unmatched_pc_trials': unmatched_pc_trials,
            'matched_platform_events': matched_platform_events,
            'unmatched_platform_events': unmatched_platform_events,
            'pc_match_rate': len(matched_pc_trials) / len(self.pc_trials) if self.pc_trials else 0,
            'max_match_distance': max_match_distance
        }
    
    def align_and_match_linear(self, scale, offset, max_match_distance=0.5):
        """
        Apply linear transformation to PC trial times and match them to platform events.
        
        Args:
            scale (float): Scale factor for time transformation
            offset (float): Time offset to apply after scaling
            max_match_distance (float): Maximum time difference for matching (seconds)
            
        Returns:
            dict: Matching results
        """
        matched_pc_trials = []
        unmatched_pc_trials = []
        matched_platform_events = []
        unmatched_platform_events = []
        
        platform_array = np.array(self.platform_events)
        used_platform_events = set()
        
        # Match each PC trial to nearest platform event
        for i, pc_trial_time in enumerate(self.pc_trials):
            # Apply linear transformation
            adjusted_time = pc_trial_time * scale + offset
            
            # Find nearest platform event
            distances = np.abs(platform_array - adjusted_time)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance <= max_match_distance and nearest_idx not in used_platform_events:
                # Match found
                matched_pc_trials.append({
                    'pc_trial_time': pc_trial_time,
                    'adjusted_time': adjusted_time,
                    'platform_time': platform_array[nearest_idx],
                    'distance': nearest_distance,
                    'trial_idx': i
                })
                used_platform_events.add(nearest_idx)
            else:
                # No match
                unmatched_pc_trials.append({
                    'pc_trial_time': pc_trial_time,
                    'adjusted_time': adjusted_time,
                    'nearest_distance': nearest_distance,
                    'trial_idx': i
                })
        
        # Find unmatched platform events
        for i, platform_time in enumerate(self.platform_events):
            if i not in used_platform_events:
                unmatched_platform_events.append({
                    'platform_time': platform_time,
                    'platform_idx': i
                })
            else:
                matched_platform_events.append({
                    'platform_time': platform_time,
                    'platform_idx': i
                })
        
        # Calculate clock drift info
        if len(self.pc_trials) > 1:
            session_duration = self.pc_trials[-1] - self.pc_trials[0]
            drift_seconds = session_duration * (scale - 1.0)
            drift_ms_per_hour = (scale - 1.0) * 3600 * 1000  # milliseconds per hour
        else:
            session_duration = 0
            drift_seconds = 0
            drift_ms_per_hour = 0
        
        return {
            'scale': scale,
            'offset': offset,
            'matched_pc_trials': matched_pc_trials,
            'unmatched_pc_trials': unmatched_pc_trials,
            'matched_platform_events': matched_platform_events,
            'unmatched_platform_events': unmatched_platform_events,
            'pc_match_rate': len(matched_pc_trials) / len(self.pc_trials) if self.pc_trials else 0,
            'max_match_distance': max_match_distance,
            'session_duration': session_duration,
            'total_drift': drift_seconds,
            'drift_rate_ms_per_hour': drift_ms_per_hour,
            'transformation_type': 'linear'
        }
    
    def analyze_timing_differences(self, results):
        """
        Analyse the timing differences between matched PC trials and platform events after offset correction.
        
        Args:
            results (dict): Results from align_and_match
            
        Returns:
            dict: Statistics about timing differences
        """
        if not results['matched_pc_trials']:
            return {
                'mean': None,
                'std': None,
                'median': None,
                'min': None,
                'max': None,
                'rms': None,
                'count': 0,
                'signed_mean': None,
                'signed_std': None,
                'signed_median': None
            }
        
        # Extract timing differences (absolute)
        differences = [match['distance'] for match in results['matched_pc_trials']]
        differences_array = np.array(differences)
        
        # Extract signed timing differences (PC time - platform time)
        signed_differences = []
        for match in results['matched_pc_trials']:
            signed_diff = match['adjusted_time'] - match['platform_time']
            signed_differences.append(signed_diff)
        signed_differences_array = np.array(signed_differences)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(differences_array),
            'std': np.std(differences_array),
            'median': np.median(differences_array),
            'min': np.min(differences_array),
            'max': np.max(differences_array),
            'rms': np.sqrt(np.mean(differences_array**2)),
            'count': len(differences),
            'percentiles': {
                '25th': np.percentile(differences_array, 25),
                '75th': np.percentile(differences_array, 75),
                '90th': np.percentile(differences_array, 90),
                '95th': np.percentile(differences_array, 95)
            },
            # Signed statistics
            'signed_mean': np.mean(signed_differences_array),
            'signed_std': np.std(signed_differences_array),
            'signed_median': np.median(signed_differences_array),
            'signed_min': np.min(signed_differences_array),
            'signed_max': np.max(signed_differences_array),
            'signed_percentiles': {
                '5th': np.percentile(signed_differences_array, 5),
                '25th': np.percentile(signed_differences_array, 25),
                '75th': np.percentile(signed_differences_array, 75),
                '95th': np.percentile(signed_differences_array, 95)
            }
        }
        
        # Find matches within different thresholds
        stats['within_thresholds'] = {
            '100ms': np.sum(differences_array <= 0.1),
            '250ms': np.sum(differences_array <= 0.25),
            '500ms': np.sum(differences_array <= 0.5),
            '1s': np.sum(differences_array <= 1.0)
        }
        
        # Count early vs late
        stats['n_pc_early'] = np.sum(signed_differences_array < 0)
        stats['n_pc_late'] = np.sum(signed_differences_array > 0)
        stats['n_exact'] = np.sum(signed_differences_array == 0)
        
        return stats
    
    def refine_alignment_with_matches(self, initial_results, max_scale_deviation=0.01):
        """
        Perform a second alignment using only the matched trials from the first alignment
        to learn the transformation, then apply it to ALL PC trials.
        
        Args:
            initial_results (dict): Results from the first alignment
            max_scale_deviation (float): Maximum deviation from scale=1.0 (default 1%)
            
        Returns:
            dict: Refined alignment results with transformation applied to all trials
        """
        # Extract matched pairs from initial alignment
        matched_pairs = initial_results['matched_pc_trials']
        
        if len(matched_pairs) < 10:
            print("WARNING: Too few matched pairs for refinement. Returning initial results.")
            return initial_results
        
        # Create equal-length lists of matched timestamps only
        matched_pc_times = np.array([match['pc_trial_time'] for match in matched_pairs])
        matched_platform_times = np.array([match['platform_time'] for match in matched_pairs])
        
        print(f"\nRefining alignment using {len(matched_pairs)} matched pairs...")
        print(f"  Learning transformation from {len(matched_pairs)}/{len(self.pc_trials)} PC trials")
        print(f"  Will apply to all {len(self.pc_trials)} PC trials")
        
        # Fit linear transformation using least squares on matched pairs only
        slope, intercept, r_value, p_value, std_err = linregress(matched_pc_times, matched_platform_times)
        
        print(f"\nLinear fit results (from matched pairs):")
        print(f"  Scale factor: {slope:.8f}")
        print(f"  Offset: {intercept:.3f}s")
        print(f"  R-squared: {r_value**2:.6f}")
        print(f"  Clock drift: {(slope-1)*3600*1000:.3f} ms/hour")
        
        # Check if scale is within bounds
        if abs(slope - 1.0) > max_scale_deviation:
            print(f"WARNING: Scale factor {slope} exceeds maximum deviation. Using simple offset only.")
            slope = 1.0
        
        # Now apply the refined transformation to ALL PC trials
        print("\nApplying refined transformation to all PC trials...")
        refined_results = self.align_and_match_linear(slope, intercept)
        
        # Calculate improvement on the original matched pairs
        initial_rms = np.sqrt(np.mean([m['distance']**2 for m in matched_pairs]))
        
        # Find the same matched pairs in the refined results to calculate improvement
        refined_matched_distances = []
        for initial_match in matched_pairs:
            pc_time = initial_match['pc_trial_time']
            platform_time = initial_match['platform_time']
            # Find this pair in refined results
            for refined_match in refined_results['matched_pc_trials']:
                if abs(refined_match['pc_trial_time'] - pc_time) < 0.001:
                    refined_matched_distances.append(refined_match['distance'])
                    break
        
        if refined_matched_distances:
            refined_rms = np.sqrt(np.mean(np.array(refined_matched_distances)**2))
            print(f"\nAlignment improvement (on original matched pairs):")
            print(f"  Initial RMS error: {initial_rms:.3f}s")
            print(f"  Refined RMS error: {refined_rms:.3f}s")
            print(f"  Improvement: {(1 - refined_rms/initial_rms)*100:.1f}%")
        
        # Add information about the refinement process
        refined_results['refinement_info'] = {
            'n_pairs_used_for_fit': len(matched_pairs),
            'initial_offset': initial_results['offset'],
            'refined_scale': slope,
            'refined_offset': intercept,
            'r_squared': r_value**2
        }
        
        return refined_results
    
    def refine_with_best_matches(self, results, threshold_ms=80):
        """
        Perform a third alignment using only matches within threshold_ms of the mean distance.
        
        Args:
            results (dict): Results from previous alignment
            threshold_ms (float): Threshold in milliseconds (default 80ms)
            
        Returns:
            dict: Final refined alignment results
        """
        # Get timing statistics
        timing_stats = self.analyze_timing_differences(results)
        
        if timing_stats['mean'] is None:
            print("WARNING: No matched pairs available for refinement.")
            return results
        
        threshold_seconds = threshold_ms / 1000.0
        mean_distance = timing_stats['mean']
        
        # Select best matches within threshold of mean
        best_matches = []
        for match in results['matched_pc_trials']:
            if abs(match['distance'] - mean_distance) <= threshold_seconds:
                best_matches.append(match)
        
        if len(best_matches) < 5:
            print(f"WARNING: Only {len(best_matches)} matches within {threshold_ms}ms of mean. Too few for refinement.")
            return results
        
        print(f"\nFinal refinement using {len(best_matches)} best matches (within {threshold_ms}ms of mean {mean_distance*1000:.1f}ms)")
        
        # Extract timestamps from best matches
        best_pc_times = np.array([match['pc_trial_time'] for match in best_matches])
        best_platform_times = np.array([match['platform_time'] for match in best_matches])
        
        # Fit linear transformation using best matches only
        slope, intercept, r_value, p_value, std_err = linregress(best_pc_times, best_platform_times)
        
        print(f"\nFinal linear fit results (from best matches):")
        print(f"  Scale factor: {slope:.8f}")
        print(f"  Offset: {intercept:.3f}s")
        print(f"  R-squared: {r_value**2:.6f}")
        print(f"  Clock drift: {(slope-1)*3600*1000:.3f} ms/hour")
        
        # Apply final transformation to ALL PC trials
        print("\nApplying final transformation to all PC trials...")
        final_results = self.align_and_match_linear(slope, intercept)
        
        # Add information about the refinement process
        final_results['final_refinement_info'] = {
            'n_best_matches': len(best_matches),
            'threshold_ms': threshold_ms,
            'mean_distance_before': mean_distance,
            'final_scale': slope,
            'final_offset': intercept,
            'r_squared': r_value**2
        }
        
        # Copy previous refinement info if it exists
        if 'refinement_info' in results:
            final_results['refinement_info'] = results['refinement_info']
        
        return final_results
    
    def get_aligned_trial_times(self, results, final_correction=0.002):
        """
        Get all PC trial times with their aligned (DAQ) times.
        
        Args:
            results (dict): Final alignment results
            final_correction (float): Additional correction in seconds (default 0.002 for 2ms)
            
        Returns:
            list: List of dicts with trial information and aligned times
        """
        aligned_trials = []
        
        # Get transformation parameters
        if 'scale' in results:
            scale = results['scale']
            offset = results['offset']
        else:
            scale = 1.0
            offset = results['offset']
        
        # Apply transformation to all PC trials
        for i, (pc_time, trial_info) in enumerate(zip(self.pc_trials, self.pc_trials_full)):
            # Apply transformation plus final correction
            aligned_time = pc_time * scale + offset + final_correction
            
            # Find if this trial was matched
            matched = False
            match_distance = None
            for match in results['matched_pc_trials']:
                if match['trial_idx'] == i:
                    matched = True
                    match_distance = match['distance']
                    break
            
            aligned_trials.append({
                'trial_idx': i,
                'pc_time': pc_time,
                'aligned_time': aligned_time,
                'port': trial_info['port'],
                'outcome': trial_info['outcome'],
                'complete': trial_info['complete'],
                'matched': matched,
                'match_distance': match_distance
            })
        
        return aligned_trials
    
    def run_three_stage_alignment(self):
        """
        Run a three-stage alignment process:
        1. Initial alignment with simple offset
        2. Refined alignment using matched pairs with linear transformation
        3. Final refinement using only best matches
        
        Returns:
            tuple: (initial_results, refined_results, final_results, aligned_trials)
        """
        print("\n" + "="*60)
        print("THREE-STAGE ALIGNMENT PROCESS")
        print("="*60)
        
        # Load data
        print("\nLoading scales data...")
        self.load_scales_data()
        
        # Get platform events
        platform_times = self.get_platform_events(min_duration=1.0)
        print(f"Found {len(platform_times)} platform events (≥1s duration)")
        
        # Load PC trials
        print("\nLoading PC trials...")
        trials = self.load_pc_trials()
        print(f"Found {len(trials)} PC trials")
        
        # Show breakdown by port
        port_counts = {}
        for trial in trials:
            port = trial.get('port', 'unknown')
            port_counts[port] = port_counts.get(port, 0) + 1
        
        print("\nTrials by port:")
        for port in sorted(port_counts.keys()):
            print(f"  Port {port}: {port_counts[port]}")
        
        # Stage 1: Initial alignment
        print("\n" + "-"*60)
        print("STAGE 1: Initial alignment with simple offset")
        print("-"*60)
        
        initial_offset = self.find_time_offset_minimize()
        print(f"Best offset: {initial_offset:.3f} seconds")
        
        initial_results = self.align_and_match(initial_offset)
        initial_stats = self.analyze_timing_differences(initial_results)
        initial_results['timing_stats'] = initial_stats
        
        print(f"\nInitial Alignment Results:")
        print(f"  Matched PC trials: {len(initial_results['matched_pc_trials'])}/{len(self.pc_trials)} ({initial_results['pc_match_rate']*100:.1f}%)")
        if initial_stats['mean'] is not None:
            print(f"  Mean timing difference: {initial_stats['mean']*1000:.1f}ms")
            print(f"  RMS error: {initial_stats['rms']*1000:.1f}ms")
        
        # Stage 2: Refined alignment
        print("\n" + "-"*60)
        print("STAGE 2: Refined alignment with matched pairs")
        print("-"*60)
        
        refined_results = self.refine_alignment_with_matches(initial_results)
        refined_stats = self.analyze_timing_differences(refined_results)
        refined_results['timing_stats'] = refined_stats
        
        print(f"\nRefined Alignment Results:")
        print(f"  Matched PC trials: {len(refined_results['matched_pc_trials'])}/{len(self.pc_trials)} ({refined_results['pc_match_rate']*100:.1f}%)")
        if refined_stats['mean'] is not None:
            print(f"  Mean timing difference: {refined_stats['mean']*1000:.1f}ms")
            print(f"  RMS error: {refined_stats['rms']*1000:.1f}ms")
        
        # Stage 3: Final refinement with best matches
        print("\n" + "-"*60)
        print("STAGE 3: Final refinement with best matches")
        print("-"*60)
        
        final_results = self.refine_with_best_matches(refined_results, threshold_ms=80)
        final_stats = self.analyze_timing_differences(final_results)
        final_results['timing_stats'] = final_stats
        
        print(f"\nFinal Alignment Results:")
        print(f"  Matched PC trials: {len(final_results['matched_pc_trials'])}/{len(self.pc_trials)} ({final_results['pc_match_rate']*100:.1f}%)")
        
        if final_stats['mean'] is not None:
            print(f"\nFinal Timing Statistics:")
            print(f"  Mean difference: {final_stats['mean']*1000:.1f}ms")
            print(f"  Std deviation: {final_stats['std']*1000:.1f}ms")
            print(f"  Median difference: {final_stats['median']*1000:.1f}ms")
            print(f"  RMS error: {final_stats['rms']*1000:.1f}ms")
            print(f"  Range: [{final_stats['min']*1000:.1f}, {final_stats['max']*1000:.1f}]ms")
            
            print(f"\nMatches within thresholds:")
            for threshold, count in final_stats['within_thresholds'].items():
                percentage = (count / final_stats['count']) * 100 if final_stats['count'] > 0 else 0
                print(f"  Within {threshold}: {count}/{final_stats['count']} ({percentage:.1f}%)")
        
        # Get aligned trial times with final 2ms correction
        aligned_trials = self.get_aligned_trial_times(final_results, final_correction=0.002)
        
        # Print summary of transformations
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY")
        print("="*60)
        print(f"Stage 1 - Simple offset: {initial_results['offset']:.3f}s")
        if 'refinement_info' in final_results:
            print(f"Stage 2 - Linear transformation:")
            print(f"  Scale: {final_results['refinement_info']['refined_scale']:.8f}")
            print(f"  Offset: {final_results['refinement_info']['refined_offset']:.3f}s")
        if 'final_refinement_info' in final_results:
            print(f"Stage 3 - Final transformation:")
            print(f"  Scale: {final_results['final_refinement_info']['final_scale']:.8f}")
            print(f"  Offset: {final_results['final_refinement_info']['final_offset']:.3f}s")
            print(f"  Clock drift: {(final_results['final_refinement_info']['final_scale']-1)*3600*1000:.3f} ms/hour")
            print(f"  Final correction: +2ms (empirical adjustment)")
        
        # Print sample of aligned trial data
        print("\n" + "="*60)
        print("ALIGNED TRIAL DATA (first 10 trials)")
        print("="*60)
        print(f"{'Idx':>4} {'PC Time':>10} {'Aligned':>10} {'Port':>5} {'Outcome':>10} {'Matched':>8} {'Distance':>10}")
        print("-" * 60)
        
        for trial in aligned_trials[:10]:
            matched_str = "Yes" if trial['matched'] else "No"
            distance_str = f"{trial['match_distance']*1000:.1f}ms" if trial['match_distance'] is not None else "N/A"
            port_str = str(trial['port']) if trial['port'] is not None else "None"
            outcome_str = trial['outcome'] if trial['outcome'] is not None else "None"
            print(f"{trial['trial_idx']:>4} {trial['pc_time']:>10.3f} {trial['aligned_time']:>10.3f} "
                  f"{port_str:>5} {outcome_str:>10} {matched_str:>8} {distance_str:>10}")
        
        if len(aligned_trials) > 10:
            print(f"... ({len(aligned_trials) - 10} more trials)")
        
        return initial_results, refined_results, final_results, aligned_trials
    
    def get_aligned_trials_quiet(self):
        """
        Run the three-stage alignment process without printing output.
        Returns only the aligned trials list.
        """
        # Load data
        self.load_scales_data()
        self.get_platform_events(min_duration=1.0)
        self.load_pc_trials()
        
        # Stage 1: Initial alignment
        initial_offset = self.find_time_offset_minimize()
        initial_results = self.align_and_match(initial_offset)
        
        # Stage 2: Refined alignment
        refined_results = self.refine_alignment_with_matches(initial_results)
        
        # Stage 3: Final refinement
        final_results = self.refine_with_best_matches(refined_results, threshold_ms=80)
        
        # Get aligned trial times with final 2ms correction
        aligned_trials = self.get_aligned_trial_times(final_results, final_correction=0.002)
        
        return aligned_trials


# Example usage
if __name__ == "__main__":
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    
    print("Loading cohort info...")
    cohort = Cohort_folder('/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics')
    
    session_dict = cohort.get_session("250624_143350_mtao101-3d")
    
    # Create aligner
    aligner = ScalesTrialAligner(session_dict)
    
    # Run three-stage alignment
    initial_results, refined_results, final_results, aligned_trials = aligner.run_three_stage_alignment()
    
    # The aligned_trials list now contains all PC trials with their aligned DAQ times
    print(f"\n\nTotal aligned trials available: {len(aligned_trials)}")