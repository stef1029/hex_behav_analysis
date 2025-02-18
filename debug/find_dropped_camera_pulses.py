import json
from pathlib import Path
import traceback
from datetime import datetime
import numpy as np
import h5py
from pynwb import NWBHDF5IO

# For color-coded console output, especially for dropped pulses
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

# -------------------------------------------------------------------------
# 1) Your original Cohort_folder class and imports go here
#    (exact code from your script with no changes)
# -------------------------------------------------------------------------
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

from utils.plot_graphical_cohort_info import graphical_cohort_info
from utils.Cohort_folder import Cohort_folder

# -------------------------------------------------------------------------
# 2) Dropped-pulses detection functions (with debug prints)
# -------------------------------------------------------------------------
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

def count_missed_pulses(signal, timestamps, threshold=0.5, debug=False):
    """
    Detect pulses and estimate how many might have been missed by comparing
    intervals to the typical interval (median).

    :param signal: 1D array-like of signal values (camera TTL).
    :param timestamps: 1D array-like of timestamps.
    :param threshold: Rising-edge detection threshold.
    :param debug: If True, prints info about each interval ratio & how many pulses might be missed.
    :return: (total_detected, total_missed)
    """
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    total_detected = len(edges)

    if total_detected < 2:
        return total_detected, 0

    # intervals is the difference in edge indices (i.e. sample index intervals).
    # If you want time intervals, you'd do np.diff(timestamps[edges]) instead.
    intervals = np.diff(edges)
    typical_interval = np.median(intervals)

    missed_pulses = 0
    if debug:
        print(f"    Detected {total_detected} edges. Intervals between edges:")
        print(f"    Median interval = {typical_interval:.2f}")

    for i, delta in enumerate(intervals):
        ratio = delta / typical_interval
        missed_here = int(np.floor(ratio + 0.5)) - 1
        if missed_here < 0:
            missed_here = 0
        
        # Print debug info for intervals that are big enough to suspect missing pulses
        if debug and ratio >= 1.5:  # or some other threshold you prefer
            print(f"      Interval {i}: value={delta}, ratio={ratio:.2f}, "
                  f"=> missed {missed_here} pulses here.")
        
        missed_pulses += missed_here
    
    return total_detected, missed_pulses

def check_early_pulses(signal, timestamps, threshold=0.5, min_start_delay=0.027, debug=False):
    """
    Check if the first detected pulse (train of pulses) starts too early.
    By default, we require at least ~27 ms of delay (0.027 seconds) from the
    beginning of the recording until the first pulse. If the first pulse time
    is below this threshold, we report a warning.

    :param signal: 1D array-like of signal values (camera TTL).
    :param timestamps: 1D array-like of timestamps.
    :param threshold: Rising-edge detection threshold for pulses.
    :param min_start_delay: Minimal acceptable delay from t=0 to first pulse in seconds.
    :param debug: If True, prints debug statements.
    """
    edges = detect_pulse_edges(signal, timestamps, threshold=threshold)
    if len(edges) == 0:
        if debug:
            print("  No pulses found, skipping early pulse check.")
        return

    # first pulse index -> time
    first_pulse_time = timestamps[edges[0]]
    if debug:
        print(f"  First pulse at t={first_pulse_time:.4f} s. Checking if < {min_start_delay:.4f} s.")

    # Compare to the threshold
    if first_pulse_time < min_start_delay:
        print(f"  {Fore.RED}WARNING: The first pulse occurs too early (t={first_pulse_time*1000:.2f} ms), "
              f"which is less than the expected {min_start_delay*1000:.2f} ms delay.{Style.RESET_ALL}")
    else:
        if debug:
            print(f"  The first pulse occurs at {first_pulse_time:.4f} s, which is above the minimum "
                  f"required delay of {min_start_delay:.4f} s.")
        
# -------------------------------------------------------------------------
# 3) Function to iterate over all sessions in the cohort and perform checks
# -------------------------------------------------------------------------
def analyze_cohort_for_dropped_pulses(cohort_obj, camera_channel_name="CAMERA", threshold=0.5, debug=False):
    """
    For each session in the cohort:
      - Looks up the original HDF5 (arduino_DAQ_h5) to detect pulses in 'camera_channel_name'
      - Estimates dropped pulses
      - Checks if any pulses start too early in the recording
      - Opens the NWB file to count how many frames are in 'behaviour_video'
      - Prints the results, with colorization for missed pulses.
      - If debug=True, prints detailed info about intervals and ratio-based missed pulses detection.
    """
    cohort_dict = cohort_obj.cohort  # The dictionary with all mice and sessions

    for mouse_id, mouse_data in cohort_dict["mice"].items():
        for session_id, session_info in mouse_data["sessions"].items():
            print(f"\n--- Mouse: {mouse_id}, Session: {session_id} ---")

            # 1) Get HDF5 path
            h5_path = None
            raw_data = session_info.get("raw_data", {})
            if raw_data:
                h5_path = raw_data.get("arduino_DAQ_h5", None)
            
            if not h5_path or h5_path == "None":
                print("  No valid ArduinoDAQ.h5 file found. Skipping pulse detection.")
                continue

            # 2) Detect pulses from the original HDF5 file
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    timestamps = h5f['timestamps'][:]
                    channel_group = h5f['channel_data']
                    if camera_channel_name not in channel_group:
                        print(f"  Channel '{camera_channel_name}' not found in {h5_path}. Skipping.")
                        continue
                    camera_signal = channel_group[camera_channel_name][:]
                
                detected, missed = count_missed_pulses(
                    camera_signal, 
                    timestamps, 
                    threshold=threshold, 
                    debug=debug
                )
                
                # Check for early pulses
                check_early_pulses(
                    camera_signal,
                    timestamps,
                    threshold=threshold,
                    min_start_delay=0.050,  # default 50 ms
                    debug=debug
                )

                print(f"  H5 File: {h5_path}")
                print(f"    Total pulses detected : {detected}")

                # Color the missed pulses: red if missed > 0, green if 0
                if missed > 0:
                    missed_pulses_str = f"{Fore.RED}{missed}{Style.RESET_ALL}"
                else:
                    missed_pulses_str = f"{Fore.GREEN}{missed}{Style.RESET_ALL}"

                print(f"    Estimated missed pulses: {missed_pulses_str}")

            except Exception as e:
                print(f"  Error reading {h5_path}: {str(e)}")
                continue

            # 3) Open NWB file to count frames in 'behaviour_video'
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

# -------------------------------------------------------------------------
# 4) Main script entry point
# -------------------------------------------------------------------------
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

    # Now analyze for dropped pulses across the entire cohort,
    # enable debug=True for detailed interval info
    analyze_cohort_for_dropped_pulses(
        cohort_obj=cohort,
        camera_channel_name="CAMERA",
        threshold=0.5,
        debug=True   # Turn on to see interval ratios and missed pulses details
    )

if __name__ == "__main__":
    main()
