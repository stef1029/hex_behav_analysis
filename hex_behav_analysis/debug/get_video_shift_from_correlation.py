import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import json
import warnings
import traceback


def get_brightness_sensor_alignment(session_dir, sensor_name="SENSOR1", max_lag_seconds=15.0, 
                                   save_plots=False, verbose=False):
    """
    Determine the alignment/shift between brightness data and sensor data for a given session.
    
    Args:
        session_dir (str): Directory containing the session data
        sensor_name (str, optional): Name of the sensor channel to use. Defaults to "SENSOR1".
        max_lag_seconds (float, optional): Maximum time lag to search in seconds. Defaults to 15.0.
        save_plots (bool, optional): Whether to save alignment plots. Defaults to False.
        verbose (bool, optional): Whether to print detailed progress information. Defaults to False.
        
    Returns:
        float or None: The best lag in seconds, or None if alignment failed
    """
    # Set up warnings filter
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Extract session ID from directory path
    session_id = os.path.basename(session_dir)
    
    if verbose:
        print(f"Processing session: {session_id}")
        print(f"Session directory: {session_dir}")
    
    # Check if session directory exists
    if not os.path.exists(session_dir):
        if verbose:
            print(f"Session directory does not exist: {session_dir}")
        return None
    
    # Find brightness data file
    truncated_start_dir = os.path.join(session_dir, "truncated_start_report")
    if not os.path.exists(truncated_start_dir):
        if verbose:
            print(f"Truncated start report directory not found: {truncated_start_dir}")
        return None
    
    brightness_files = [f for f in os.listdir(truncated_start_dir) 
                       if f.endswith("_brightness_data.csv") and session_id in f]
    
    if not brightness_files:
        # Try looking for any brightness file if session_id-specific one not found
        brightness_files = [f for f in os.listdir(truncated_start_dir) 
                           if f.endswith("_brightness_data.csv")]
        
    if not brightness_files:
        if verbose:
            print(f"No brightness data file found in: {truncated_start_dir}")
        return None
    
    brightness_file = os.path.join(truncated_start_dir, brightness_files[0])
    
    # Find Arduino DAQ file
    arduino_daq_files = [f for f in os.listdir(session_dir) if f.endswith("-ArduinoDAQ.h5")]
    if not arduino_daq_files:
        # Try with different naming convention
        arduino_daq_files = [f for f in os.listdir(session_dir) if f.endswith(".h5") and "ArduinoDAQ" in f]
    
    if not arduino_daq_files:
        if verbose:
            print(f"No Arduino DAQ h5 file found in: {session_dir}")
        return None
    
    arduino_daq_file = os.path.join(session_dir, arduino_daq_files[0])
    
    # Find tracker data JSON file
    tracker_data_files = [f for f in os.listdir(session_dir) if f.endswith("_Tracker_data.json")]
    if not tracker_data_files:
        if verbose:
            print(f"No Tracker data JSON file found in: {session_dir}")
        return None
    
    tracker_data_file = os.path.join(session_dir, tracker_data_files[0])
    
    if verbose:
        print(f"Found Arduino DAQ file: {arduino_daq_file}")
        print(f"Found tracker data file: {tracker_data_file}")
        print(f"Found brightness file: {brightness_file}")
    
    # Extract camera frame times from Arduino DAQ and tracker data
    frame_times, video_timestamps_df = extract_camera_frame_times(arduino_daq_file, tracker_data_file, verbose)
    
    if frame_times is None or video_timestamps_df is None or video_timestamps_df.empty:
        if verbose:
            print("Failed to extract camera frame times")
        return None
    
    # Extract sensor data from the Arduino DAQ file
    sensor_df = extract_sensor_data_from_daq(arduino_daq_file, sensor_name, verbose)
    
    if sensor_df is None or sensor_df.empty:
        if verbose:
            print(f"Failed to extract {sensor_name} data from Arduino DAQ")
        return None
    
    # Perform alignment
    output_dir = truncated_start_dir if save_plots else None
    lag, correlation = align_brightness_with_sensor(
        brightness_file, sensor_df, video_timestamps_df, 
        output_dir, max_lag_seconds, verbose=verbose
    )
    
    if verbose and lag is not None:
        print(f"Alignment complete. Lag: {lag:.3f}s, Correlation: {correlation:.3f}")
    
    return lag


def extract_camera_frame_times(arduino_daq_file, tracker_data_file, verbose=False):
    """
    Extract camera frame times from Arduino DAQ h5 file and tracker data JSON.
    
    Args:
        arduino_daq_file: Path to Arduino DAQ h5 file
        tracker_data_file: Path to tracker data JSON file
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (frame_times_dict, video_timestamps_df)
    """
    try:
        # Load the Arduino DAQ h5 file
        with h5py.File(arduino_daq_file, 'r') as h5f:
            # First check the structure of the h5 file
            if verbose:
                print("HDF5 file structure:")
                for key in h5f.keys():
                    print(f"  - {key}")
                    if isinstance(h5f[key], h5py.Group):
                        for subkey in h5f[key].keys():
                            print(f"    - {subkey}")
            
            # Extract camera data and timestamps
            if 'channel_data' in h5f and 'CAMERA' in h5f['channel_data']:
                camera_data = np.array(h5f['channel_data']['CAMERA'])
                if 'timestamps' in h5f:
                    timestamps = np.array(h5f['timestamps'])
                else:
                    if verbose:
                        print("No timestamps found in h5 file")
                    return None, None
            else:
                if verbose:
                    print("Could not find CAMERA data in the expected structure")
                return None, None
        
        # Detect low-to-high transitions (0 -> 1) in camera data
        low_to_high_transitions = np.where((camera_data[:-1] == 0) & (camera_data[1:] == 1))[0]
        
        # Get the timestamps corresponding to these transitions
        camera_pulses = timestamps[low_to_high_transitions + 1]  # Add 1 for the high point
        
        # Import tracker data JSON
        with open(tracker_data_file, 'r') as f:
            video_metadata = json.load(f)
        
        # Check if frame_IDs is a key
        if "frame_IDs" in video_metadata:
            frame_IDs = video_metadata["frame_IDs"]
        else:
            if verbose:
                print("Error: No frame_IDs found in tracker_data.json")
            return None, None
        
        # Create pulse_times dictionary
        pulse_times = {i: pulse for i, pulse in enumerate(camera_pulses)}
        
        # Determine how many valid frames we can process
        max_available_frame = min(len(camera_pulses) - 1, max(frame_IDs))
        
        # If we have fewer camera pulses than frame IDs, truncate the frame IDs
        if max_available_frame < max(frame_IDs):
            truncated_frame_ids = [f for f in frame_IDs if f <= max_available_frame]
            frame_IDs = truncated_frame_ids
            if verbose:
                print(
                    f"Warning: Only {len(camera_pulses)} camera pulses recorded, "
                    f"so truncating frame IDs to {len(frame_IDs)}."
                )
        
        # Create frame_times dictionary
        frame_times = {}
        for frame_ID in frame_IDs:
            if frame_ID in pulse_times:
                frame_times[frame_ID] = pulse_times[frame_ID]
            elif frame_ID < len(pulse_times):
                frame_times[frame_ID] = pulse_times[frame_ID]
        
        # Create DataFrame for video timestamps
        video_timestamps_df = pd.DataFrame({
            'frame': list(frame_times.keys()),
            'timestamp': list(frame_times.values())
        })
        
        if verbose:
            print(f"Extracted {len(frame_times)} frame timestamps")
        
        return frame_times, video_timestamps_df
        
    except Exception as e:
        if verbose:
            print(f"Error extracting camera frame times: {str(e)}")
            traceback.print_exc()
        return None, None


def extract_sensor_data_from_daq(arduino_daq_file, sensor_name="SENSOR1", verbose=False):
    """
    Extract sensor data from Arduino DAQ h5 file.
    
    Args:
        arduino_daq_file: Path to Arduino DAQ h5 file
        sensor_name: Name of the sensor channel
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with sensor data
    """
    try:
        with h5py.File(arduino_daq_file, 'r') as h5f:
            # Check if sensor channel exists in the expected structure
            if 'channel_data' not in h5f or sensor_name not in h5f['channel_data']:
                if verbose:
                    print(f"Sensor {sensor_name} not found in Arduino DAQ file")
                return None
            
            # Load sensor channel data
            sensor_data = np.array(h5f['channel_data'][sensor_name])
            
            # Load timestamps
            if 'timestamps' not in h5f:
                if verbose:
                    print("No timestamps found in Arduino DAQ file")
                return None
                
            timestamps = np.array(h5f['timestamps'])
            
            # Detect transitions in sensor data
            # Start of nosepoke: 1->0 transition (active low)
            # End of nosepoke: 0->1 transition
            start_nosepoke = np.where((sensor_data[:-1] == 1) & (sensor_data[1:] == 0))[0]
            end_nosepoke = np.where((sensor_data[:-1] == 0) & (sensor_data[1:] == 1))[0]
            
            # Create sensor events DataFrame
            events = []
            
            # Add start events (-1 event value as in the original code)
            for idx in start_nosepoke:
                if idx + 1 < len(timestamps):
                    events.append({
                        'timestamp': timestamps[idx + 1],
                        'event': -1  # -1 means start of nosepoke
                    })
            
            # Add end events (1 event value as in the original code)
            for idx in end_nosepoke:
                if idx + 1 < len(timestamps):
                    events.append({
                        'timestamp': timestamps[idx + 1],
                        'event': 1  # 1 means end of nosepoke
                    })
            
            # Sort events by timestamp
            sensor_df = pd.DataFrame(events).sort_values('timestamp').reset_index(drop=True)
            
            if verbose:
                print(f"Extracted {len(sensor_df)} sensor events from {sensor_name}")
            
            return sensor_df
            
    except Exception as e:
        if verbose:
            print(f"Error extracting sensor data: {str(e)}")
            traceback.print_exc()
        return None


def convert_sensor_events_to_binary(sensor_df, time_grid):
    """
    Convert sensor events (start/end of nosepoke) to a binary signal on a uniform time grid.
    
    Args:
        sensor_df: DataFrame with sensor timestamps and events
        time_grid: Uniform time grid to interpolate to
        
    Returns:
        Binary array (0/1) indicating nosepoke state at each time point
    """
    if sensor_df is None or sensor_df.empty:
        return None
    
    # Initialize binary signal (0 = no nosepoke, 1 = nosepoke)
    binary_signal = np.zeros_like(time_grid, dtype=int)
    
    # In the DAQ_to_nwb code we see that for SENSOR channels:
    # - HIGH=0 was used (meaning active-low)
    # - -1 events mean "start of nosepoke"
    # - 1 events mean "end of nosepoke"
    
    # Track the current state (0 = no nosepoke, 1 = nosepoke)
    current_state = 0
    
    # Sort events by timestamp to ensure chronological processing
    sorted_events = sensor_df.sort_values('timestamp')
    
    for _, row in sorted_events.iterrows():
        # Find the index in time_grid that's closest to this timestamp
        idx = np.searchsorted(time_grid, row['timestamp'])
        if idx >= len(time_grid):
            idx = len(time_grid) - 1
        
        # Update state based on the event
        if row['event'] == -1:  # Start of nosepoke
            current_state = 1
        elif row['event'] == 1:  # End of nosepoke
            current_state = 0
        
        # Update all subsequent values
        binary_signal[idx:] = current_state
    
    return binary_signal


def align_brightness_with_sensor(brightness_file, sensor_df, video_timestamps_df, output_dir=None, 
                               max_lag_seconds=5.0, verbose=False):
    """
    Align video brightness data with sensor data using cross-correlation.
    
    Args:
        brightness_file: Path to the brightness CSV file
        sensor_df: DataFrame with sensor timestamps and events
        video_timestamps_df: DataFrame with video frame timestamps
        output_dir: Directory to save the alignment plots (if None, plots won't be saved)
        max_lag_seconds: Maximum time lag to search (in seconds)
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (lag_in_seconds, correlation_score)
    """
    try:
        # 1. Load brightness data
        brightness_df = pd.read_csv(brightness_file)
        if verbose:
            print(f"Loaded brightness data with columns: {brightness_df.columns.tolist()}")
        
        # 2. Determine which columns to use
        # For brightness file: if there are 3 columns, assume frame, timestamp_ms, brightness
        # If 2 columns, assume frame, brightness
        if len(brightness_df.columns) >= 3:
            frame_col = brightness_df.columns[0]
            brightness_col = brightness_df.columns[2]
        elif len(brightness_df.columns) == 2:
            frame_col = brightness_df.columns[0]
            brightness_col = brightness_df.columns[1]
        else:
            if verbose:
                print(f"Unexpected columns in brightness file: {brightness_df.columns}")
            return None, 0
        
        if verbose:
            print(f"Using columns: frame={frame_col}, brightness={brightness_col}")
        
        # 3. Check if we have the video timestamps
        if video_timestamps_df is None:
            if verbose:
                print("No video timestamps available, can't proceed with alignment")
            return None, 0
            
        # 4. Merge brightness data with video timestamps based on frame number
        # First ensure frame columns have the same data type
        brightness_df[frame_col] = brightness_df[frame_col].astype(int)
        video_timestamps_df['frame'] = video_timestamps_df['frame'].astype(int)
        
        # Merge to get actual timestamps for each brightness value
        merged_df = pd.merge(
            brightness_df,
            video_timestamps_df,
            left_on=frame_col,
            right_on='frame',
            how='inner'
        )
        
        if merged_df.empty:
            if verbose:
                print("Could not merge brightness data with video timestamps, frame numbers don't match")
            # Try a different approach - match by row index if frames are sequential
            if len(brightness_df) <= len(video_timestamps_df):
                if verbose:
                    print("Trying to match by index instead...")
                merged_df = brightness_df.copy()
                merged_df['timestamp'] = video_timestamps_df['timestamp'].values[:len(brightness_df)]
            else:
                return None, 0
        
        # 5. Calculate threshold for brightness
        brightness_mean = merged_df[brightness_col].mean()
        brightness_std = merged_df[brightness_col].std()
        
        # Keep original threshold direction (below mean-std)
        threshold = brightness_mean - brightness_std
        
        # 6. Create binary brightness signal (1 when below threshold)
        # IMPORTANT: But INVERT the binary result (0→1, 1→0) to match sensor signal
        merged_df['binary'] = (merged_df[brightness_col] < threshold).astype(int)
        
        # 7. Create a uniform time grid for both signals
        # Find the overlapping time range
        if len(merged_df) == 0 or len(sensor_df) == 0:
            if verbose:
                print("Not enough data in either brightness or sensor data")
            return None, 0
            
        brightness_min_time = merged_df['timestamp'].min()
        brightness_max_time = merged_df['timestamp'].max()
        sensor_min_time = sensor_df['timestamp'].min()
        sensor_max_time = sensor_df['timestamp'].max()
        
        overall_min_time = max(brightness_min_time, sensor_min_time)
        overall_max_time = min(brightness_max_time, sensor_max_time)
        
        if overall_max_time <= overall_min_time:
            if verbose:
                print(f"No overlapping time range between brightness and sensor")
            return None, 0
        
        # Create a uniform time grid (100 Hz should be sufficient for accurate alignment)
        time_spacing = 0.01  # 10 ms, 100 Hz
        time_grid = np.arange(overall_min_time, overall_max_time + time_spacing, time_spacing)
        
        # 8. Interpolate brightness data to this uniform grid
        brightness_binary = merged_df['binary'].values
        brightness_times = merged_df['timestamp'].values
        
        # Interpolate brightness signal to uniform grid (nearest neighbor to preserve binary nature)
        interp_brightness = np.zeros_like(time_grid, dtype=int)
        for i, t in enumerate(time_grid):
            # Find closest timestamp
            closest_idx = np.abs(brightness_times - t).argmin()
            interp_brightness[i] = brightness_binary[closest_idx]
        
        # 9. Convert sensor events to binary signal on the same time grid
        interp_sensor = convert_sensor_events_to_binary(sensor_df, time_grid)
        
        if interp_sensor is None:
            if verbose:
                print("Failed to convert sensor events to binary signal")
            return None, 0
        
        # 10. Perform cross-correlation with specified lag range
        max_lag_points = int(max_lag_seconds / time_spacing)
        lags = np.arange(-max_lag_points, max_lag_points + 1)
        
        # Calculate cross-correlation
        # Use 'full' mode to get correlation at each lag, then extract relevant portion
        correlation = signal.correlate(interp_sensor, interp_brightness, mode='full')
        full_lags = signal.correlation_lags(len(interp_sensor), len(interp_brightness), mode='full')
        
        # Extract the portion corresponding to our lag range
        valid_indices = (full_lags >= -max_lag_points) & (full_lags <= max_lag_points)
        valid_lags = full_lags[valid_indices]
        valid_correlation = correlation[valid_indices]
        
        # 11. Find the maximum correlation
        max_corr_idx = np.argmax(valid_correlation)
        best_lag_points = valid_lags[max_corr_idx]
        best_lag_seconds = best_lag_points * time_spacing
        max_corr_value = valid_correlation[max_corr_idx]
        
        if verbose:
            print(f"Best lag: {best_lag_points} samples ({best_lag_seconds:.3f} seconds)")
            print(f"Maximum correlation: {max_corr_value:.3f}")
        
        # 12. Create and save alignment plot if requested
        if output_dir is not None:
            plt.figure(figsize=(12, 12))
            
            # Plot correlation results
            plt.subplot(411)
            plt.plot(valid_lags * time_spacing, valid_correlation)
            plt.axvline(x=best_lag_seconds, color='r', linestyle='--')
            plt.title(f'Cross-Correlation (Peak at {best_lag_seconds:.3f}s)')
            plt.xlabel('Time Lag (s)')
            plt.ylabel('Correlation')
            plt.grid(True, alpha=0.3)
            
            # Plot original brightness signal
            plt.subplot(412)
            plt.plot(merged_df['timestamp'], merged_df['binary'], '-', markersize=2, label='Video (Binary, Inverted)')
            plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.title('Binary Brightness Signal (1 = NOT Below Threshold, Inverted)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot original sensor signal (as rendered on time grid)
            plt.subplot(413)
            plt.plot(time_grid, interp_sensor, '-', color='orange', label='Sensor (Binary)')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.title('Sensor Signal (1 = Nosepoke)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot aligned signals
            plt.subplot(414)
            
            # Shift the time axis of one signal to align
            shifted_grid = time_grid + best_lag_seconds
            
            # Find the overlapping time range after shift
            overlap_min = max(time_grid[0], shifted_grid[0])
            overlap_max = min(time_grid[-1], shifted_grid[-1])
            
            # Get the portions of each signal that overlap after shifting
            brightness_mask = (shifted_grid >= overlap_min) & (shifted_grid <= overlap_max)
            sensor_mask = (time_grid >= overlap_min) & (time_grid <= overlap_max)
            
            # Plot aligned signals
            plt.plot(shifted_grid[brightness_mask], interp_brightness[brightness_mask], 
                     label='Brightness (shifted)', color='blue')
            plt.plot(time_grid[sensor_mask], interp_sensor[sensor_mask], 
                     label='Sensor', color='orange', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.title('Aligned Signals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            file_base = os.path.basename(brightness_file).replace('_brightness_data.csv', '')
            output_file = os.path.join(output_dir, f"{file_base}_sensor_alignment.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"Alignment plot saved to: {output_file}")
            
            # Save alignment data
            alignment_data = pd.DataFrame([{
                'session_id': file_base,
                'lag_samples': best_lag_points,
                'lag_seconds': best_lag_seconds,
                'correlation_score': max_corr_value,
                'brightness_min_time': brightness_min_time,
                'brightness_max_time': brightness_max_time,
                'sensor_min_time': sensor_min_time,
                'sensor_max_time': sensor_max_time,
                'overlap_min_time': overall_min_time,
                'overlap_max_time': overall_max_time,
                'threshold_value': threshold,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std
            }])
            
            csv_path = os.path.join(output_dir, f"{file_base}_alignment_info.csv")
            alignment_data.to_csv(csv_path, index=False)
            
            if verbose:
                print(f"Alignment information saved to: {csv_path}")
        
        return best_lag_seconds, max_corr_value
        
    except Exception as e:
        if verbose:
            print(f"Error processing {brightness_file}: {str(e)}")
            traceback.print_exc()
        return None, 0


# Example usage:
if __name__ == "__main__":
    # Example function call
    session_dir = "/path/to/session_directory"
    
    lag = get_brightness_sensor_alignment(
        session_dir=session_dir,
        sensor_name="SENSOR1",
        max_lag_seconds=15.0,
        save_plots=True,
        verbose=True
    )
    
    session_id = os.path.basename(session_dir)
    if lag is not None:
        print(f"Session {session_id} has a brightness-sensor lag of {lag:.3f} seconds")
    else:
        print(f"Failed to determine lag for session {session_id}")