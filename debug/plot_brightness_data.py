import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import h5py
from pynwb import NWBHDF5IO
import warnings
import traceback

def extract_data_from_nwb(nwb_file_path, sensor_name="SENSOR1"):
    """
    Extract both sensor data and video timestamps from NWB file.
    
    Args:
        nwb_file_path: Path to NWB file
        sensor_name: Name of the sensor channel (default: "SENSOR1")
        
    Returns:
        Tuple of (sensor_df, video_timestamps_df)
    """
    try:
        with NWBHDF5IO(nwb_file_path, 'r') as io:
            nwbfile = io.read()
            
            # 1. Extract sensor data
            sensor_df = None
            if sensor_name in nwbfile.acquisition:
                sensor_data = nwbfile.acquisition[sensor_name]
                
                # Get sensor timestamps and intervals
                sensor_timestamps = sensor_data.timestamps[:]
                sensor_intervals = sensor_data.data[:]
                
                # Create dataframe with sensor events
                sensor_df = pd.DataFrame({
                    'timestamp': sensor_timestamps,
                    'event': sensor_intervals
                })
                
                print(f"  Extracted {len(sensor_timestamps)} sensor events from {sensor_name}")
            else:
                print(f"  Sensor {sensor_name} not found in NWB file")
            
            # 2. Extract video frame timestamps
            video_timestamps_df = None
            video_timestamps = None
            
            # Try to find the behavior video in acquisitions
            for acq_name in nwbfile.acquisition:
                if 'video' in acq_name.lower() or 'behaviour' in acq_name.lower() or 'behavior' in acq_name.lower():
                    video_series = nwbfile.acquisition[acq_name]
                    if hasattr(video_series, 'timestamps'):
                        video_timestamps = video_series.timestamps[:]
                        print(f"  Found video timestamps in '{acq_name}' with {len(video_timestamps)} frames")
                        
                        # Create dataframe with frame numbers and timestamps
                        video_timestamps_df = pd.DataFrame({
                            'frame': np.arange(len(video_timestamps)),
                            'timestamp': video_timestamps
                        })
                        break
            
            if video_timestamps_df is None:
                print("  No video timestamps found in NWB file")
                
            return sensor_df, video_timestamps_df
            
    except Exception as e:
        print(f"  Error reading NWB file {nwb_file_path}: {str(e)}")
        traceback.print_exc()
        return None, None

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

def align_brightness_with_sensor(brightness_file, sensor_df, video_timestamps_df, output_dir, max_lag_seconds=5.0):
    """
    Align video brightness data with sensor data using cross-correlation.
    
    Args:
        brightness_file: Path to the brightness CSV file
        sensor_df: DataFrame with sensor timestamps and events
        video_timestamps_df: DataFrame with video frame timestamps
        output_dir: Directory to save the alignment plots
        max_lag_seconds: Maximum time lag to search (in seconds)
        
    Returns:
        Tuple of (lag_in_seconds, correlation_score)
    """
    try:
        # 1. Load brightness data
        brightness_df = pd.read_csv(brightness_file)
        print(f"  Loaded brightness data with columns: {brightness_df.columns.tolist()}")
        
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
            print(f"  Unexpected columns in brightness file: {brightness_df.columns}")
            return None, 0
        
        print(f"  Using columns: frame={frame_col}, brightness={brightness_col}")
        
        # 3. Check if we have the video timestamps
        if video_timestamps_df is None:
            print("  No video timestamps available from NWB, can't proceed with alignment")
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
            print("  Could not merge brightness data with video timestamps, frame numbers don't match")
            # Try a different approach - match by row index if frames are sequential
            if len(brightness_df) <= len(video_timestamps_df):
                print("  Trying to match by index instead...")
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
        merged_df['binary'] = 1 - (merged_df[brightness_col] < threshold).astype(int)
        
        # 7. Create a uniform time grid for both signals
        # Find the overlapping time range
        if len(merged_df) == 0 or len(sensor_df) == 0:
            print("  Not enough data in either brightness or sensor data")
            return None, 0
            
        brightness_min_time = merged_df['timestamp'].min()
        brightness_max_time = merged_df['timestamp'].max()
        sensor_min_time = sensor_df['timestamp'].min()
        sensor_max_time = sensor_df['timestamp'].max()
        
        overall_min_time = max(brightness_min_time, sensor_min_time)
        overall_max_time = min(brightness_max_time, sensor_max_time)
        
        if overall_max_time <= overall_min_time:
            print(f"  No overlapping time range between brightness ({brightness_min_time:.2f}-{brightness_max_time:.2f}) and sensor ({sensor_min_time:.2f}-{sensor_max_time:.2f})")
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
            print("  Failed to convert sensor events to binary signal")
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
        
        print(f"  Best lag: {best_lag_points} samples ({best_lag_seconds:.3f} seconds)")
        print(f"  Maximum correlation: {max_corr_value:.3f}")
        
        # 12. Create alignment plot
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
        
        print(f"  Alignment plot saved to: {output_file}")
        
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
        print(f"  Alignment information saved to: {csv_path}")
        
        return best_lag_seconds, max_corr_value
        
    except Exception as e:
        print(f"  Error processing {brightness_file}: {str(e)}")
        traceback.print_exc()
        return None, 0

def find_brightness_files_with_nwb(cohort_dir):
    """
    Find all brightness data files with corresponding NWB files.
    
    Args:
        cohort_dir: Path to the cohort directory
        
    Returns:
        List of tuples containing (session_id, brightness_file_path, nwb_file_path)
    """
    file_pairs = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(cohort_dir):
        # Check if this is a truncated_start_report folder
        if os.path.basename(root) == "truncated_start_report":
            # Look for brightness data files
            brightness_files = [f for f in files if f.endswith("_brightness_data.csv")]
            
            for brightness_file in brightness_files:
                # Extract session ID from filename
                session_id = brightness_file.replace("_brightness_data.csv", "")
                
                # The session directory is the parent directory of truncated_start_report
                session_dir = os.path.dirname(root)
                session_name = os.path.basename(session_dir)
                
                # Look for NWB files in the session directory
                nwb_files = [f for f in os.listdir(session_dir) if f.endswith(".nwb")]
                
                if nwb_files:
                    nwb_file = os.path.join(session_dir, nwb_files[0])
                    print(f"Found matching NWB file: {nwb_file}")
                    file_pairs.append((session_id, os.path.join(root, brightness_file), nwb_file))
                else:
                    print(f"No NWB file found in session directory: {session_dir}")
                    
    return file_pairs

def main():
    """
    Main function with hardcoded parameters.
    Modify these values directly instead of using command line arguments.
    """
    # Set your parameters here
    cohort_dir = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"  # Path to your cohort directory
    sensor_name = "SENSOR1"  # Name of the sensor to extract from NWB
    max_lag_seconds = 15.0  # Maximum time lag to search in seconds
    
    # Find all brightness data files with corresponding NWB files
    file_pairs = find_brightness_files_with_nwb(cohort_dir)
    
    if file_pairs:
        print(f"Found {len(file_pairs)} pairs of brightness and NWB files")
        
        # Create a summary dataframe
        summary_data = []
        
        for session_id, brightness_file, nwb_file in file_pairs:
            print(f"\nProcessing {session_id}...")
            print(f"  Brightness: {brightness_file}")
            print(f"  NWB: {nwb_file}")
            
            # Extract sensor data and video timestamps from NWB
            sensor_df, video_timestamps_df = extract_data_from_nwb(nwb_file, sensor_name)
            
            if sensor_df is not None and not sensor_df.empty and video_timestamps_df is not None and not video_timestamps_df.empty:
                print(f"  Extracted sensor data with {len(sensor_df)} events and video timestamps for {len(video_timestamps_df)} frames")
                
                # Align brightness with sensor data
                output_dir = os.path.dirname(brightness_file)
                lag, correlation = align_brightness_with_sensor(
                    brightness_file, 
                    sensor_df, 
                    video_timestamps_df, 
                    output_dir,
                    max_lag_seconds
                )
                
                if lag is not None:
                    summary_data.append({
                        'session_id': session_id,
                        'lag_seconds': lag,
                        'correlation': correlation
                    })
                    print(f"  Alignment complete. Lag: {lag:.3f}s, Correlation: {correlation:.3f}")
                else:
                    print(f"  Failed to align brightness and sensor data")
            else:
                print(f"  Failed to extract required data from NWB")
        
        if summary_data:
            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(cohort_dir, "brightness_sensor_alignment_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to: {summary_path}")
            
            # Calculate statistics
            mean_lag = summary_df['lag_seconds'].mean()
            std_lag = summary_df['lag_seconds'].std()
            mean_corr = summary_df['correlation'].mean()
            
            print(f"\nSummary statistics:")
            print(f"  Mean lag: {mean_lag:.3f}s ± {std_lag:.3f}s")
            print(f"  Mean correlation: {mean_corr:.3f}")
            
            # Create a summary plot
            plt.figure(figsize=(10, 6))
            plt.subplot(211)
            plt.bar(summary_df['session_id'], summary_df['lag_seconds'])
            plt.title('Time Lag by Session')
            plt.ylabel('Lag (seconds)')
            plt.xticks(rotation=90)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(212)
            plt.bar(summary_df['session_id'], summary_df['correlation'])
            plt.title('Correlation Score by Session')
            plt.ylabel('Correlation')
            plt.xticks(rotation=90)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            summary_plot_path = os.path.join(cohort_dir, "brightness_sensor_alignment_summary.png")
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Summary plot saved to: {summary_plot_path}")
        else:
            print("\nNo successful alignments to summarize")
    else:
        print("No matching file pairs found. Please check the cohort directory and file structure.")
        
        # Option for manual file specification
        # Uncomment and customize these lines if needed
        # manual_brightness_file = r"/path/to/specific/brightness_data.csv"
        # manual_nwb_file = r"/path/to/specific/session.nwb"
        # manual_output_dir = os.path.dirname(manual_brightness_file)
        
        # print(f"\nProcessing manual files...")
        # sensor_df, video_timestamps_df = extract_data_from_nwb(manual_nwb_file, sensor_name)
        # if sensor_df is not None and not sensor_df.empty and video_timestamps_df is not None:
        #     lag, correlation = align_brightness_with_sensor(
        #         manual_brightness_file, sensor_df, video_timestamps_df, manual_output_dir, max_lag_seconds
        #     )
        #     print(f"Manual alignment complete. Lag: {lag:.3f}s, Correlation: {correlation:.3f}")

if __name__ == "__main__":
    # Suppress FutureWarnings that often come from h5py/NWB libraries
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()