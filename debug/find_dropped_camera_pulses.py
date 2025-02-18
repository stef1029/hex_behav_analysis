import json
from pathlib import Path
import traceback
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

def detect_pulse_edges(signal, timestamps, threshold=0.5):
    """
    Identify the indices of rising edges (TTL pulses) in the given signal.
    A rising edge is where signal[i] < threshold and signal[i+1] >= threshold.
    """
    signal = np.asarray(signal)
    if len(signal) == 0:
        return np.array([])

    # Above/below threshold
    above_thresh = signal >= threshold
    # A rising edge is a transition from "below threshold" to "above threshold"
    rising = (above_thresh[1:] == True) & (above_thresh[:-1] == False)
    
    edges = np.where(rising)[0] + 1  # +1 to offset because we shifted by 1
    return edges

def count_pulse_anomalies(signal, timestamps, threshold=0.5, min_ratio=0.5, debug=False):
    """
    Detect pulses and estimate anomalies: both missed pulses and pulses that are too close together.
    
    :param signal: 1D array-like of signal values (camera TTL)
    :param timestamps: 1D array-like of timestamps
    :param threshold: Rising-edge detection threshold
    :param min_ratio: Minimum acceptable ratio of interval/typical_interval
    :param debug: If True, prints info about each interval ratio
    :return: (total_detected, total_missed, too_close_count, too_close_locations)
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
        print(f"    Detected {total_detected} edges. Analyzing intervals:")
        print(f"    Median interval = {typical_interval:.2f}")

    for i, delta in enumerate(intervals):
        ratio = delta / typical_interval
        
        # Check for missed pulses (intervals too long)
        missed_here = int(np.floor(ratio + 0.5)) - 1
        if missed_here < 0:
            missed_here = 0
        missed_pulses += missed_here
        
        # Check for pulses too close together (intervals too short)
        if ratio < min_ratio:
            too_close_count += 1
            # Store the timestamp for this anomaly
            pulse_time = timestamps[edges[i]]
            too_close_locations.append((pulse_time, ratio))
            
            if debug:
                print(f"      {Fore.YELLOW}Pulses too close at t={pulse_time:.4f}s: "
                      f"interval={delta}, ratio={ratio:.3f}{Style.RESET_ALL}")
        
        # Print debug info for any anomalous intervals
        elif debug and (ratio >= 1.5 or ratio < min_ratio):
            print(f"      Interval {i}: value={delta}, ratio={ratio:.2f}, "
                  f"missed pulses={missed_here}")
    
    return total_detected, missed_pulses, too_close_count, too_close_locations

def check_early_pulses(signal, timestamps, threshold=0.5, min_start_delay=0.027, debug=False):
    """
    Check if the first detected pulse starts too early.
    
    :param signal: 1D array-like of signal values (camera TTL)
    :param timestamps: 1D array-like of timestamps
    :param threshold: Rising-edge detection threshold for pulses
    :param min_start_delay: Minimal acceptable delay from t=0 to first pulse in seconds
    :param debug: If True, prints debug statements
    """
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    if len(edges) == 0:
        if debug:
            print("  No pulses found, skipping early pulse check.")
        return

    first_pulse_time = timestamps[edges[0]]
    if debug:
        print(f"  First pulse at t={first_pulse_time:.4f} s. Checking if < {min_start_delay:.4f} s.")

    if first_pulse_time < min_start_delay:
        print(f"  {Fore.RED}WARNING: The first pulse occurs too early (t={first_pulse_time*1000:.2f} ms), "
              f"which is less than the expected {min_start_delay*1000:.2f} ms delay.{Style.RESET_ALL}")
    else:
        if debug:
            print(f"  The first pulse occurs at {first_pulse_time:.4f} s, which is above the minimum "
                  f"required delay of {min_start_delay:.4f} s.")

def analyze_cohort_for_dropped_pulses(cohort_obj, camera_channel_name="CAMERA", threshold=0.5, min_ratio=0.5, debug=False):
    """
    Analyze each session in the cohort for pulse anomalies.
    """
    cohort_dict = cohort_obj.cohort

    for mouse_id, mouse_data in cohort_dict["mice"].items():
        for session_id, session_info in mouse_data["sessions"].items():
            print(f"\n--- Mouse: {mouse_id}, Session: {session_id} ---")

            # Get HDF5 path
            h5_path = None
            raw_data = session_info.get("raw_data", {})
            if raw_data:
                h5_path = raw_data.get("arduino_DAQ_h5", None)
            
            if not h5_path or h5_path == "None":
                print("  No valid ArduinoDAQ.h5 file found. Skipping pulse detection.")
                continue

            try:
                with h5py.File(h5_path, 'r') as h5f:
                    timestamps = h5f['timestamps'][:]
                    channel_group = h5f['channel_data']
                    if camera_channel_name not in channel_group:
                        print(f"  Channel '{camera_channel_name}' not found in {h5_path}. Skipping.")
                        continue
                    camera_signal = channel_group[camera_channel_name][:]
                
                detected, missed, too_close, close_locations = count_pulse_anomalies(
                    camera_signal, 
                    timestamps, 
                    threshold=threshold,
                    min_ratio=min_ratio,
                    debug=debug
                )
                
                # Check for early pulses
                check_early_pulses(
                    camera_signal,
                    timestamps,
                    threshold=threshold,
                    min_start_delay=0.050,
                    debug=debug
                )

                print(f"  H5 File: {h5_path}")
                print(f"    Total pulses detected : {detected}")

                # Color code the anomalies
                if missed > 0:
                    missed_str = f"{Fore.RED}{missed}{Style.RESET_ALL}"
                else:
                    missed_str = f"{Fore.GREEN}{missed}{Style.RESET_ALL}"

                if too_close > 0:
                    too_close_str = f"{Fore.YELLOW}{too_close}{Style.RESET_ALL}"
                else:
                    too_close_str = f"{Fore.GREEN}{too_close}{Style.RESET_ALL}"

                print(f"    Estimated missed pulses: {missed_str}")
                print(f"    Pulses too close together: {too_close_str}")

                # Print details of pulses that were too close
                if too_close > 0:
                    print("    Details of pulses too close together:")
                    for time, ratio in close_locations:
                        print(f"      t={time:.4f}s, ratio={ratio:.3f}")

            except Exception as e:
                print(f"  Error reading {h5_path}: {str(e)}")
                continue

            # Check NWB file for frame count
            nwb_path = session_info['processed_data'].get("NWB_file", "None")
            if not nwb_path or nwb_path == "None":
                print("  No NWB file found for this session. Skipping video frame count.")
                continue

            try:
                with NWBHDF5IO(str(nwb_path), 'r') as io:
                    nwbfile = io.read()
                    if 'behaviour_video' in nwbfile.acquisition:
                        video_obj = nwbfile.acquisition['behaviour_video']
                        if video_obj.timestamps is not None:
                            num_video_frames = len(video_obj.timestamps)
                            print(f"  NWB File: {nwb_path}")
                            print(f"    Number of frames in 'behaviour_video': {num_video_frames}")
                        else:
                            print("    'behaviour_video' found, but timestamps were None.")
                    else:
                        print("    No 'behaviour_video' in NWB acquisition.")
            except Exception as e:
                print(f"  Error reading NWB file {nwb_path}: {str(e)}")

def main():
    # Path to your cohort folder (update as needed)
    cohort_dir = r"/cephfs2/dwelch/Behaviour/November_cohort"

    # Create or load the cohort info
    cohort = Cohort_folder(
        cohort_directory=cohort_dir,
        multi=True,
        portable_data=False,
        OEAB_legacy=False,
        ignore_tests=True,
        use_existing_cohort_info=False,
        plot=False
    )

    # Analyze for dropped pulses across the entire cohort
    analyze_cohort_for_dropped_pulses(
        cohort_obj=cohort,
        camera_channel_name="CAMERA",
        threshold=0.5,
        min_ratio=0.5,  # Intervals shorter than half the typical interval will be flagged
        debug=True      # Turn on to see interval ratios and missed pulses details
    )

if __name__ == "__main__":
    main()