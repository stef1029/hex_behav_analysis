import json
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
from pynwb import NWBHDF5IO

# For color-coded console output
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

from utils.plot_graphical_cohort_info import graphical_cohort_info
from utils.Cohort_folder import Cohort_folder

# Color scheme for output
class Colors:
    HEADER = Fore.CYAN
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    INFO = Fore.WHITE
    HIGHLIGHT = Fore.MAGENTA
    
def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{text}")
    print(f"{Colors.HEADER}{'=' * len(text)}{Style.RESET_ALL}")

def print_status(category, value, threshold=None, is_good=None):
    """Print a status line with appropriate coloring."""
    if is_good is None and threshold is not None:
        is_good = value <= threshold
    
    color = Colors.SUCCESS if is_good else Colors.WARNING if value > 0 else Colors.SUCCESS
    print(f"  {Colors.INFO}{category:<25}: {color}{value}{Style.RESET_ALL}")

def detect_pulse_edges(signal, timestamps, threshold=0.5):
    """
    Identify the indices of rising edges (TTL pulses) in the given signal.
    A rising edge is where signal[i] < threshold and signal[i+1] >= threshold.
    """
    signal = np.asarray(signal)
    if len(signal) == 0:
        return np.array([])

    above_thresh = signal >= threshold
    rising = (above_thresh[1:] == True) & (above_thresh[:-1] == False)
    edges = np.where(rising)[0] + 1
    return edges

def count_pulse_anomalies(signal, timestamps, threshold=0.5, min_ratio=0.5, debug=False):
    """
    Detect pulses and estimate anomalies: both missed pulses and pulses that are too close together.
    """
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    total_detected = len(edges)

    if total_detected < 2:
        return total_detected, 0, 0, []

    intervals = np.diff(edges)
    typical_interval = np.median(intervals)

    missed_pulses = 0
    too_close_count = 0
    too_close_locations = []

    if debug:
        print(f"{Colors.INFO}    Detected {total_detected} edges. Median interval = {typical_interval:.2f}")

    for i, delta in enumerate(intervals):
        ratio = delta / typical_interval
        
        # Check for missed pulses
        missed_here = int(np.floor(ratio + 0.5)) - 1
        if missed_here > 0:
            missed_pulses += missed_here
        
        # Check for pulses too close together
        if ratio < min_ratio:
            too_close_count += 1
            pulse_time = timestamps[edges[i]]
            too_close_locations.append((float(pulse_time), float(ratio)))
            
            if debug:
                print(f"    {Colors.WARNING}Pulses too close at t={pulse_time:.4f}s "
                      f"(interval ratio={ratio:.3f}){Style.RESET_ALL}")
    
    return total_detected, missed_pulses, too_close_count, too_close_locations

def check_early_pulses(signal, timestamps, threshold=0.5, min_start_delay=0.027):
    """Check if the first detected pulse starts too early."""
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    if len(edges) == 0:
        return None
    
    first_pulse_time = timestamps[edges[0]]
    return first_pulse_time < min_start_delay, float(first_pulse_time)

def check_late_pulses(signal, timestamps, threshold=0.5, min_end_delay=0.1):
    """Check if the last detected pulse ends too early before the end of recording."""
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    if len(edges) == 0:
        return None
    
    last_pulse_time = timestamps[edges[-1]]
    recording_end_time = timestamps[-1]
    end_delay = recording_end_time - last_pulse_time
    return end_delay < min_end_delay, float(end_delay)

def check_for_crash_pattern(session_dir):
    """
    Check Tracker_data.json for signs of a crashed session.
    Returns tuple of (is_crash_pattern, total_frames, details).
    """
    try:
        # Find the Tracker_data.json file
        json_files = list(Path(session_dir).glob("*Tracker_data.json"))
        if not json_files:
            return False, None, "No Tracker_data.json file found"
            
        # Read the JSON file
        with open(json_files[0], 'r') as f:
            tracker_data = json.load(f)
            
        # Get frame IDs
        frame_ids = tracker_data.get('frame_IDs', [])
        if not frame_ids:
            return False, None, "No frame_IDs found in Tracker_data.json"
            
        total_frames = len(frame_ids)
        
        # Check if total frames is a multiple of 200
        is_multiple_of_200 = (total_frames % 200 == 0)
        
        details = (
            f"Total frames: {total_frames}"
            f" (multiple of 200: {is_multiple_of_200})"
        )
        
        return is_multiple_of_200, total_frames, details
            
    except Exception as e:
        return False, None, f"Error reading Tracker_data.json: {str(e)}"

def analyze_session_pulses(h5_path, nwb_path, session_dir, camera_channel_name="CAMERA", threshold=0.5, 
                           min_ratio=0.5, debug=False):
    """Analyze pulse data for a single session."""
    results = {
        'status': 'success',
        'messages': [],
        'data': {},
        'crash_info': None
    }
    
    # Check for crash pattern in Tracker_data.json
    is_crash, total_frames, crash_details = check_for_crash_pattern(session_dir)
    results['crash_info'] = {
        'is_crash_pattern': is_crash,
        'total_frames': total_frames,
        'details': crash_details
    }
    
    try:
        # Read HDF5 data
        with h5py.File(h5_path, 'r') as h5f:
            timestamps = h5f['timestamps'][:]
            if camera_channel_name not in h5f['channel_data']:
                results['status'] = 'error'
                results['messages'].append(f"Channel '{camera_channel_name}' not found")
                return results
            
            camera_signal = h5f['channel_data'][camera_channel_name][:]
        
        # Analyze pulses
        detected, missed, too_close, close_locations = count_pulse_anomalies(
            camera_signal, timestamps, threshold=threshold, min_ratio=min_ratio, debug=debug
        )
        
        # Check timing
        early_result = check_early_pulses(camera_signal, timestamps, threshold=threshold)
        late_result = check_late_pulses(camera_signal, timestamps, threshold=threshold)
        
        # Store results
        results['data'].update({
            'total_pulses': detected,
            'missed_pulses': missed,
            'too_close_pulses': too_close,
            'close_locations': close_locations,
            'early_pulse': early_result[0] if early_result else None,
            'first_pulse_time': early_result[1] if early_result else None,
            'late_pulse': late_result[0] if late_result else None,
            'end_delay': late_result[1] if late_result else None
        })
        
        # Check NWB file
        if nwb_path and nwb_path != "None":
            try:
                with NWBHDF5IO(str(nwb_path), 'r') as io:
                    nwbfile = io.read()
                    if 'behaviour_video' in nwbfile.acquisition:
                        video_obj = nwbfile.acquisition['behaviour_video']
                        if video_obj.timestamps is not None:
                            results['data']['video_frames'] = len(video_obj.timestamps)
            except Exception as e:
                results['messages'].append(f"NWB error: {str(e)}")
        
    except Exception as e:
        results['status'] = 'error'
        results['messages'].append(str(e))
    
    return results

def save_problem_sessions(problem_sessions, output_file):
    """
    Save information about problematic sessions to a file.
    Only saves sessions that have actual issues.
    """
    with open(output_file, 'w') as f:
        f.write("Problem Sessions Report\n")
        f.write("=====================\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for session in problem_sessions:
            if session['has_issues']:  # Only write sessions with actual problems
                f.write(f"\nMouse: {session['mouse_id']}\n")
                f.write(f"Session: {session['session_id']}\n")
                f.write(f"File: {session['h5_path']}\n")
                f.write("Issues:\n")
                
                # Write specific issues
                if session['missed_pulses'] > 0:
                    f.write(f"- Missed pulses: {session['missed_pulses']}\n")
                if session['too_close_pulses'] > 0:
                    f.write(f"- Too-close pulses: {session['too_close_pulses']}\n")
                    f.write("  Locations:\n")
                    for time, ratio in session['close_locations']:
                        f.write(f"    t={time:.4f}s, ratio={ratio:.3f}\n")
                if session['early_pulse']:
                    f.write(f"- First pulse too early: {session['first_pulse_time']*1000:.1f}ms\n")
                if session['late_pulse']:
                    f.write(f"- Last pulse too close to end: {session['end_delay']*1000:.1f}ms\n")
                if session.get('crash_info', {}).get('is_crash_pattern'):
                    f.write("- POTENTIAL CRASH DETECTED:\n")
                    f.write(f"  {session['crash_info']['details']}\n")
                if session.get('error_messages'):
                    f.write("- Errors:\n")
                    for msg in session['error_messages']:
                        f.write(f"  {msg}\n")
                f.write("-" * 50 + "\n")

def analyze_cohort_for_dropped_pulses(cohort_obj, camera_channel_name="CAMERA", threshold=0.5, 
                                      min_ratio=0.5, debug=False):
    """Analyze pulse anomalies across the entire cohort and produce final summary."""
    problem_sessions = []  # List to store sessions with issues
    cohort_dict = cohort_obj.cohort
    
    # Counters to produce final summary
    summary_counters = {
        'total_sessions': 0,    # total valid sessions (with h5 to analyze)
        'crashes': 0,
        'truncated_starts': 0,  # early pulses
        'truncated_ends': 0,    # late pulses
        'both_truncated': 0
    }
    
    for mouse_id, mouse_data in cohort_dict["mice"].items():
        for session_id, session_info in mouse_data["sessions"].items():
            print_header(f"Mouse: {mouse_id}, Session: {session_id}")
            
            # Get file paths and directory
            h5_path = session_info.get("raw_data", {}).get("arduino_DAQ_h5", None)
            nwb_path = session_info['processed_data'].get("NWB_file", "None")
            session_dir = Path(h5_path).parent if h5_path and h5_path != "None" else None
            
            if not h5_path or h5_path == "None":
                print(f"{Colors.WARNING}  No valid ArduinoDAQ.h5 file found. Skipping.{Style.RESET_ALL}")
                continue  # Skip to next session
            
            # We have a valid H5 path, so increment total_sessions
            summary_counters['total_sessions'] += 1
            
            # Analyze session
            results = analyze_session_pulses(
                h5_path, nwb_path, session_dir, camera_channel_name, threshold, min_ratio, debug
            )
            
            if results['status'] == 'error':
                print(f"{Colors.ERROR}  Analysis failed: {', '.join(results['messages'])}{Style.RESET_ALL}")
                continue
            
            # Print results
            data = results['data']
            print(f"{Colors.INFO}  File: {Path(h5_path).name}")
            
            print_status("Total pulses detected", data['total_pulses'])
            print_status("Missed pulses", data['missed_pulses'])
            print_status("Too-close pulses", data['too_close_pulses'])
            
            if data['early_pulse']:
                print(f"  {Colors.WARNING}WARNING: First pulse too early (t={data['first_pulse_time']*1000:.1f}ms)")
            if data['late_pulse']:
                print(f"  {Colors.WARNING}WARNING: Last pulse too close to end (delay={data['end_delay']*1000:.1f}ms)")
            
            if data.get('video_frames'):
                print(f"  {Colors.INFO}Video frames in NWB: {data['video_frames']}")
            
            if data['too_close_pulses'] > 0:
                print(f"\n  {Colors.WARNING}Pulses too close together:{Style.RESET_ALL}")
                for time, ratio in data['close_locations']:
                    print(f"    t={time:.4f}s, ratio={ratio:.3f}")
            
            # Print crash pattern info if detected
            if results['crash_info']['is_crash_pattern']:
                print(f"  {Colors.ERROR}POTENTIAL CRASH DETECTED: {results['crash_info']['details']}{Style.RESET_ALL}")
            
            if results['messages']:
                print(f"\n  {Colors.INFO}Additional messages:")
                for msg in results['messages']:
                    print(f"    {msg}")
            
            # Check if this session had any issues
            has_issues = (
                data['missed_pulses'] > 0 or 
                data['too_close_pulses'] > 0 or 
                data['early_pulse'] or 
                data['late_pulse'] or
                results['status'] == 'error' or
                results['crash_info']['is_crash_pattern']
            )
            
            # Update summary counters if appropriate
            if results['crash_info']['is_crash_pattern']:
                summary_counters['crashes'] += 1
            if data['early_pulse']:
                summary_counters['truncated_starts'] += 1
            if data['late_pulse']:
                summary_counters['truncated_ends'] += 1
            if data['early_pulse'] and data['late_pulse']:
                summary_counters['both_truncated'] += 1
            
            # If truncated start, create a subdirectory and store a JSON file with session data
            if data['early_pulse']:
                report_dir = session_dir / "truncated_start_report"
                report_dir.mkdir(exist_ok=True)  # Make directory if it doesn't exist
                
                # Build the JSON report data
                report_data = {
                    "mouse_id": mouse_id,
                    "session_id": session_id,
                    "h5_path": str(h5_path),
                    "truncated_start": True,
                    "first_pulse_time_ms": data['first_pulse_time'] * 1000 if data['first_pulse_time'] else None,
                    "missed_pulses": data['missed_pulses'],
                    "too_close_pulses": data['too_close_pulses'],
                    "too_close_locations": data['close_locations'],
                    "late_pulse_detected": bool(data['late_pulse']),
                    "crash_detected": results['crash_info']['is_crash_pattern'],
                    "crash_details": results['crash_info']['details'],
                    "messages": results['messages']
                }
                
                # Save the JSON file
                report_file = report_dir / "truncated_start_info.json"
                with open(report_file, "w") as f:
                    json.dump(report_data, f, indent=4)
            
            # If there were any issues, add to problem sessions list
            if has_issues:
                problem_sessions.append({
                    'mouse_id': mouse_id,
                    'session_id': session_id,
                    'h5_path': h5_path,
                    'has_issues': has_issues,
                    'missed_pulses': data['missed_pulses'],
                    'too_close_pulses': data['too_close_pulses'],
                    'close_locations': data['close_locations'],
                    'early_pulse': data['early_pulse'],
                    'first_pulse_time': data['first_pulse_time'],
                    'late_pulse': data['late_pulse'],
                    'end_delay': data['end_delay'],
                    'error_messages': results['messages'],
                    'crash_info': results['crash_info']
                })
    
    # Save problem sessions to file if any were found
    if problem_sessions:
        output_file = Path(cohort_obj.cohort_directory) / "problem_sessions.txt"
        save_problem_sessions(problem_sessions, output_file)
        print(f"\n{Colors.HIGHLIGHT}Problem sessions report saved to: {output_file}{Style.RESET_ALL}")
    
    # Final summary report
    print_header("FINAL SUMMARY REPORT")
    print(f"{Colors.INFO}Total sessions analyzed: {summary_counters['total_sessions']}")
    print(f"{Colors.INFO}Sessions with crash pattern: {summary_counters['crashes']}")
    print(f"{Colors.INFO}Sessions with truncated start (early pulse): {summary_counters['truncated_starts']}")
    print(f"{Colors.INFO}Sessions with truncated end (late pulse): {summary_counters['truncated_ends']}")
    print(f"{Colors.INFO}Sessions with both truncated start and end: {summary_counters['both_truncated']}")

def main():
    # Path to cohort folder
    # cohort_dir = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"
    cohort_dir = r"/cephfs2/dwelch/Behaviour/November_cohort"

    # Create or load cohort info
    print_header("Initializing Cohort Analysis")
    cohort = Cohort_folder(
        cohort_directory=cohort_dir,
        multi=True,
        portable_data=False,
        OEAB_legacy=False,
        ignore_tests=True,
        use_existing_cohort_info=False,
        plot=False
    )

    # Analyze dropped pulses
    print_header("Starting Pulse Analysis")
    analyze_cohort_for_dropped_pulses(
        cohort_obj=cohort,
        camera_channel_name="CAMERA",
        threshold=0.5,
        min_ratio=0.5,
        debug=True
    )

if __name__ == "__main__":
    main()
