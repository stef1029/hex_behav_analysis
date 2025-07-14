"""
Script to analyse scales pulse frequency from a single H5 file and visualise frequency over time.
Uses Cohort_folder to retrieve file paths from session IDs.
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import socket
import os
import json
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder


def calculate_instantaneous_frequency(pulse_times, window_size=10):
    """
    Calculate instantaneous frequency using a sliding window approach.
    
    Parameters
    ----------
    pulse_times : array-like
        Array of pulse timestamps in seconds.
    window_size : int, optional
        Number of pulses to use for each frequency calculation (default is 10).
    
    Returns
    -------
    tuple
        (time_points, frequencies) where time_points are the centres of each window
        and frequencies are the calculated frequencies in Hz.
    """
    if len(pulse_times) < window_size + 1:
        # If not enough pulses, use all available
        intervals = np.diff(pulse_times)
        frequencies = 1.0 / intervals
        time_points = pulse_times[:-1] + intervals / 2
        return time_points, frequencies
    
    time_points = []
    frequencies = []
    
    for i in range(len(pulse_times) - window_size):
        window_pulses = pulse_times[i:i + window_size + 1]
        window_duration = window_pulses[-1] - window_pulses[0]
        frequency = window_size / window_duration
        time_centre = (window_pulses[0] + window_pulses[-1]) / 2
        
        time_points.append(time_centre)
        frequencies.append(frequency)
    
    return np.array(time_points), np.array(frequencies)


def calculate_sampling_frequency(timestamps, window_size_seconds=1.0, max_points=1000):
    """
    Calculate the sampling frequency of the DAQ system over time.
    
    Parameters
    ----------
    timestamps : array-like
        Array of all timestamps from the DAQ system in seconds.
    window_size_seconds : float, optional
        Size of the sliding window in seconds (default is 1.0).
    max_points : int, optional
        Maximum number of points to calculate (default is 1000).
    
    Returns
    -------
    tuple
        (time_points, sampling_frequencies) where time_points are the centres 
        of each window and sampling_frequencies are in Hz.
    """
    time_points = []
    sampling_frequencies = []
    
    # Start from beginning and slide window
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_duration = end_time - start_time
    
    # Calculate step size to get approximately max_points
    step_size = max(window_size_seconds / 2, total_duration / max_points)
    
    current_time = start_time
    while current_time + window_size_seconds <= end_time:
        # Use searchsorted for faster window finding
        start_idx = np.searchsorted(timestamps, current_time, side='left')
        end_idx = np.searchsorted(timestamps, current_time + window_size_seconds, side='right')
        samples_in_window = end_idx - start_idx
        
        # Calculate sampling frequency for this window
        sampling_freq = samples_in_window / window_size_seconds
        time_centre = current_time + window_size_seconds / 2
        
        time_points.append(time_centre)
        sampling_frequencies.append(sampling_freq)
        
        # Move window forward
        current_time += step_size
    
    return np.array(time_points), np.array(sampling_frequencies)


def calculate_stability_metrics(time_series_data, name="Data"):
    """
    Calculate stability metrics for a time series of frequency/rate data.
    
    Parameters
    ----------
    time_series_data : array-like
        Time series data (e.g., frequencies or rates over time).
    name : str
        Name of the data for display purposes.
    
    Returns
    -------
    dict
        Dictionary containing stability metrics.
    """
    if len(time_series_data) < 2:
        return {
            'name': name,
            'mean': 0,
            'std': 0,
            'cv': 0,
            'range': 0,
            'stable': False
        }
    
    data = np.array(time_series_data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0  # Coefficient of variation in %
    data_range = np.max(data) - np.min(data)
    
    # Calculate percentiles
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    
    # Check for sudden drops or spikes
    if len(data) > 10:
        # Calculate rolling mean with window of 5
        window = 5
        rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
        deviations = np.abs(data[window-1:] - rolling_mean) / rolling_mean
        max_deviation = np.max(deviations) * 100 if len(deviations) > 0 else 0
    else:
        max_deviation = 0
    
    # Stability criteria
    is_stable = cv < 10 and max_deviation < 20  # Less than 10% CV and 20% max deviation
    
    return {
        'name': name,
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'range': data_range,
        'min': np.min(data),
        'max': np.max(data),
        'p5': p5,
        'p95': p95,
        'max_deviation': max_deviation,
        'stable': is_stable
    }


def print_stability_analysis(daq_result, pc_result):
    """
    Print stability analysis for all three frequency/rate measurements.
    
    Parameters
    ----------
    daq_result : dict
        Result from DAQ analysis.
    pc_result : dict
        Result from PC analysis.
    """
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS")
    print("=" * 60)
    
    stability_results = []
    
    # 1. DAQ Scales Pulse Frequency
    if daq_result['success'] and len(daq_result['pulse_times']) > 10:
        pulse_times = daq_result['pulse_times']
        _, pulse_frequencies = calculate_instantaneous_frequency(pulse_times, window_size=10)
        pulse_stability = calculate_stability_metrics(pulse_frequencies, "DAQ Scales Pulse Rate")
        stability_results.append(pulse_stability)
    else:
        print("\n1. DAQ Scales Pulse Rate: Insufficient data for stability analysis")
    
    # 2. DAQ Sampling Rate - USE LARGER WINDOW AND MAX POINTS
    if daq_result['success']:
        timestamps = daq_result['timestamps']
        print(f"   Processing {len(timestamps)} DAQ timestamps...")
        _, sampling_frequencies = calculate_sampling_frequency(
            timestamps, 
            window_size_seconds=5.0,  # Larger window
            max_points=500  # Limit points
        )
        daq_stability = calculate_stability_metrics(sampling_frequencies, "DAQ Sampling Rate")
        stability_results.append(daq_stability)
    else:
        print("\n2. DAQ Sampling Rate: No data available")
    
    # 3. PC Message Reception Rate - USE LARGER WINDOW
    if pc_result['success']:
        pc_timestamps = pc_result['timestamps']
        print(f"   Processing {len(pc_timestamps)} PC timestamps...")
        _, pc_message_rates = calculate_sampling_frequency(
            pc_timestamps, 
            window_size_seconds=5.0,  # Larger window
            max_points=500  # Limit points
        )
        pc_stability = calculate_stability_metrics(pc_message_rates, "PC Message Rate")
        stability_results.append(pc_stability)
    else:
        print("\n3. PC Message Rate: No data available")
    
    # Print results
    for i, result in enumerate(stability_results, 1):
        print(f"\n{i}. {result['name']}:")
        print(f"   Mean: {result['mean']:.1f} Hz")
        print(f"   Range: [{result['min']:.1f}, {result['max']:.1f}] Hz")
        print(f"   Standard Deviation: {result['std']:.2f} Hz")
        print(f"   Coefficient of Variation: {result['cv']:.1f}%")
        print(f"   95% of values between: {result['p5']:.1f} - {result['p95']:.1f} Hz")
        
        if result['max_deviation'] > 0:
            print(f"   Max deviation from rolling mean: {result['max_deviation']:.1f}%")
        
        # Stability verdict
        if result['stable']:
            print(f"   ✓ STABLE - Low variation (CV < 10%)")
        else:
            print(f"   ⚠️  UNSTABLE - High variation detected!")
            if result['cv'] > 10:
                print(f"      - Coefficient of variation {result['cv']:.1f}% exceeds 10%")
            if result['max_deviation'] > 20:
                print(f"      - Maximum deviation {result['max_deviation']:.1f}% exceeds 20%")
    
    # Overall assessment
    print("\n" + "-" * 60)
    print("OVERALL ASSESSMENT:")
    
    all_stable = all(r['stable'] for r in stability_results)
    
    if all_stable:
        print("✓ All frequencies are STABLE")
    else:
        print("⚠️  One or more frequencies show INSTABILITY:")
        for result in stability_results:
            if not result['stable']:
                print(f"   - {result['name']}: CV={result['cv']:.1f}%, "
                      f"Range={result['range']:.1f} Hz")
    
    # Specific warnings
    if len(stability_results) >= 3:
        pc_result = stability_results[2]
        if pc_result['mean'] < 25:
            print(f"\n⚠️  WARNING: PC message rate ({pc_result['mean']:.1f} Hz) "
                  f"is below expected 30 Hz!")
        
        if pc_result['cv'] > 5 and stability_results[1]['cv'] < 5:
            print("\n⚠️  PC message rate is unstable while DAQ is stable - "
                  "suggests serial communication issues")


def analyse_scales_pulses_from_behaviour_json(json_path, threshold=20.0, verbose=True):
    """
    Analyse scales message reception from behaviour data JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to the behaviour data JSON file.
    threshold : float, optional
        Weight threshold in grams (default is 20.0) - only used for information.
    verbose : bool, optional
        If True, print detailed information during analysis (default is True).
    
    Returns
    -------
    dict
        Dictionary containing:
        - success: bool indicating if analysis was successful
        - timestamps: array of all timestamps
        - weights: array of all weight values
        - message_ids: array of all message IDs
        - recording_duration: float representing total recording duration in seconds
        - pc_message_rate_hz: float representing PC logging rate
        - missing_messages: int number of missing messages
        - error: str containing error message if analysis failed
    """
    try:
        json_path = Path(json_path)
        
        if not json_path.exists():
            return {'success': False, 'error': f'File not found: {json_path}'}
        
        if verbose:
            print(f"Opening file: {json_path}")
        
        with open(json_path, 'r') as f:
            behaviour_data = json.load(f)
        
        # Get scales data from behaviour logs
        scales_data_raw = behaviour_data.get("Scales data", [])
        
        if not scales_data_raw:
            return {'success': False, 'error': 'No "Scales data" found in behaviour logs'}
        
        # Parse scales data
        timestamps = []
        weights = []
        message_ids = []
        
        for entry in scales_data_raw:
            if len(entry) >= 3:
                timestamps.append(entry[0])
                weights.append(entry[1])
                message_ids.append(entry[2])
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        weights = np.array(weights)
        message_ids = np.array(message_ids)
        
        # Get threshold from behaviour data if available
        if "Mouse weight threshold" in behaviour_data:
            threshold = float(behaviour_data["Mouse weight threshold"])
            if verbose:
                print(f"Mouse weight threshold from file: {threshold}g")
        
        recording_duration = timestamps[-1] - timestamps[0]
        pc_message_rate = len(timestamps) / recording_duration
        
        # Check for missing message IDs
        expected_ids = np.arange(message_ids[0], message_ids[-1] + 1)
        missing_ids = np.setdiff1d(expected_ids, message_ids)
        
        if verbose:
            print(f"Recording duration: {recording_duration:.2f} seconds")
            print(f"Number of PC-logged messages: {len(timestamps)}")
            print(f"PC message rate: {pc_message_rate:.1f} Hz")
            print(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}]g")
            print(f"\nMessage ID analysis:")
            print(f"  - First ID: {message_ids[0]}")
            print(f"  - Last ID: {message_ids[-1]}")
            print(f"  - Expected messages: {len(expected_ids)}")
            print(f"  - Missing messages: {len(missing_ids)}")
            if len(missing_ids) > 0:
                print(f"  - Message loss rate: {len(missing_ids)/len(expected_ids)*100:.1f}%")
        
        return {
            'success': True,
            'timestamps': timestamps,
            'weights': weights,
            'message_ids': message_ids,
            'recording_duration': recording_duration,
            'pc_message_rate_hz': pc_message_rate,
            'missing_messages': len(missing_ids),
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyse_scales_pulses(h5_path, scales_channel="SCALES", threshold=0.5, verbose=True):
    """
    Analyse scales pulses from Arduino DAQ data in a single H5 file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file containing Arduino DAQ data.
    scales_channel : str, optional
        Name of the scales channel in the HDF5 file (default is "SCALES").
    threshold : float, optional
        Threshold value for pulse detection (default is 0.5).
    verbose : bool, optional
        If True, print detailed information during analysis (default is True).
    
    Returns
    -------
    dict
        Dictionary containing:
        - success: bool indicating if analysis was successful
        - pulse_rate_hz: float representing pulse rate in Hz (None if failed)
        - total_pulses: int representing total number of pulses detected
        - median_interval: float representing median interval between pulses in seconds
        - recording_duration: float representing total recording duration in seconds
        - pulse_times: array of pulse timestamps
        - error: str containing error message if analysis failed
    """
    try:
        h5_path = Path(h5_path)
        
        if not h5_path.exists():
            return {'success': False, 'error': f'File not found: {h5_path}'}
        
        if verbose:
            print(f"Opening file: {h5_path}")
        
        with h5py.File(h5_path, 'r') as h5f:
            # Check available channels
            if verbose:
                print(f"Available channels: {list(h5f['channel_data'].keys())}")
            
            # Get timestamps
            timestamps = h5f['timestamps'][:]
            recording_duration = timestamps[-1] - timestamps[0]
            
            if scales_channel not in h5f['channel_data']:
                return {
                    'success': False, 
                    'error': f'Channel "{scales_channel}" not found. Available: {list(h5f["channel_data"].keys())}'
                }
            
            scales_signal = h5f['channel_data'][scales_channel][:]
        
        if verbose:
            print(f"Recording duration: {recording_duration:.2f} seconds")
            print(f"Signal range: [{np.min(scales_signal):.3f}, {np.max(scales_signal):.3f}]")
        
        # Detect pulses by finding rising edges
        signal = np.asarray(scales_signal)
        above_thresh = signal >= threshold
        rising = (above_thresh[1:] == True) & (above_thresh[:-1] == False)
        edges = np.where(rising)[0] + 1
        
        if verbose:
            print(f"Detected {len(edges)} pulses using threshold {threshold}")
        
        if len(edges) < 2:
            return {
                'success': True,
                'pulse_rate_hz': 0,
                'total_pulses': len(edges),
                'median_interval': None,
                'recording_duration': recording_duration,
                'pulse_times': timestamps[edges] if len(edges) > 0 else np.array([]),
                'timestamps': timestamps,
                'error': 'Insufficient pulses for rate calculation'
            }
        
        # Calculate pulse rate from median interval
        pulse_times = timestamps[edges]
        intervals = np.diff(pulse_times)
        median_interval = np.median(intervals)
        pulse_rate = 1.0 / median_interval if median_interval > 0 else 0
        
        if verbose:
            print(f"\nPulse statistics:")
            print(f"  - Total pulses: {len(edges)}")
            print(f"  - Median interval: {median_interval:.3f} seconds")
            print(f"  - Min interval: {np.min(intervals):.3f} seconds")
            print(f"  - Max interval: {np.max(intervals):.3f} seconds")
            print(f"  - Pulse rate: {pulse_rate:.2f} Hz")
        
        return {
            'success': True,
            'pulse_rate_hz': pulse_rate,
            'total_pulses': len(edges),
            'median_interval': median_interval,
            'recording_duration': recording_duration,
            'pulse_times': pulse_times,
            'timestamps': timestamps
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def plot_frequency_over_time_comparison(daq_result, pc_result, mouse_id=None, window_size=10, threshold_freq=10.0, 
                                      save=False, show=True, web_display=False, max_plot_points=5000):
    """
    Create a matplotlib figure comparing DAQ and PC scales frequency over time.
    
    Parameters
    ----------
    daq_result : dict
        Result dictionary from analyse_scales_pulses (DAQ data).
    pc_result : dict
        Result dictionary from analyse_scales_pulses_from_behaviour_json (PC data).
    mouse_id : str, optional
        Mouse ID to display in plot title.
    window_size : int, optional
        Window size for frequency calculation (default is 10).
    threshold_freq : float, optional
        Threshold frequency to show as horizontal line (default is 10.0 Hz).
    save : bool, optional
        Whether to save the plot to 'temp_output' directory (default is False).
    show : bool, optional
        Whether to display the plot (default is True).
    web_display : bool, optional
        Whether to display the plot in a web browser on localhost (default is False).
    max_plot_points : int, optional
        Maximum number of points to plot for performance (default is 5000).
    
    Returns
    -------
    None
    """
    # Configure matplotlib backend for web display if requested
    if web_display:
        matplotlib.use('webagg')
        # Prevent automatic browser opening
        plt.rcParams['webagg.open_in_browser'] = False
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 3, 1])
    
    # Plot DAQ pulse frequency and sampling rate
    if daq_result['success'] and len(daq_result['pulse_times']) > 1:
        pulse_times = daq_result['pulse_times']
        timestamps = daq_result['timestamps']
        
        print(f"Preparing DAQ plots ({len(timestamps)} samples)...")
        
        # Calculate instantaneous frequency
        time_points, frequencies = calculate_instantaneous_frequency(pulse_times, window_size)
        
        # Downsample if too many points
        if len(time_points) > max_plot_points:
            indices = np.linspace(0, len(time_points)-1, max_plot_points, dtype=int)
            time_points = time_points[indices]
            frequencies = frequencies[indices]
        
        # Calculate sampling frequency with limited points
        sample_time_points, sampling_frequencies = calculate_sampling_frequency(
            timestamps, 
            window_size_seconds=5.0,
            max_points=1000
        )
        
        # Convert to minutes
        time_minutes = time_points / 60
        sample_time_minutes = sample_time_points / 60
        
        # DAQ pulse frequency plot
        axes[0].plot(time_minutes, frequencies, 'b-', linewidth=1.5, label='Pulse frequency', alpha=0.7)
        axes[0].axhline(y=daq_result['pulse_rate_hz'], color='darkblue', linestyle='--', 
                       label=f'Median pulse freq: {daq_result["pulse_rate_hz"]:.1f} Hz', alpha=0.7)
        
        # Add DAQ sampling rate on same plot with secondary y-axis
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(sample_time_minutes, sampling_frequencies, 'g-', linewidth=1.0, 
                     label='DAQ sampling rate', alpha=0.5)
        ax0_twin.set_ylabel('DAQ Sampling Rate (Hz)', color='g')
        ax0_twin.tick_params(axis='y', labelcolor='g')
        
        axes[0].set_ylabel('Pulse Frequency (Hz)', color='b')
        axes[0].tick_params(axis='y', labelcolor='b')
        title_prefix = f'{mouse_id} - ' if mouse_id else ''
        axes[0].set_title(f'{title_prefix}DAQ Data - Pulses: {daq_result["total_pulses"]}, '
                        f'Sampling: ~{np.median(sampling_frequencies):.0f} Hz')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left')
        ax0_twin.legend(loc='upper right')
        axes[0].set_ylim(bottom=0)
    else:
        axes[0].text(0.5, 0.5, 'Insufficient DAQ data for plotting', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Plot PC MESSAGE RATE (not pulse frequency)
    if pc_result['success']:
        pc_timestamps = pc_result['timestamps']
        
        print(f"Preparing PC plots ({len(pc_timestamps)} messages)...")
        
        # Calculate PC message rate over time with limited points
        pc_message_time_points, pc_message_rates = calculate_sampling_frequency(
            pc_timestamps, 
            window_size_seconds=5.0,
            max_points=1000
        )
        pc_message_time_minutes = pc_message_time_points / 60
        
        # Main plot: PC message reception rate
        axes[1].plot(pc_message_time_minutes, pc_message_rates, 'r-', linewidth=1.5, 
                    label='Message reception rate')
        
        # Add reference lines
        axes[1].axhline(y=30, color='green', linestyle='--', linewidth=2,
                       label='Expected rate (30 Hz)')
        axes[1].axhline(y=threshold_freq, color='orange', linestyle='--', 
                       label=f'Threshold ({threshold_freq} Hz)')
        median_rate = np.median(pc_message_rates)
        axes[1].axhline(y=median_rate, color='darkred', linestyle='--', 
                       label=f'Median: {median_rate:.1f} Hz')
        
        # Highlight regions below threshold
        axes[1].fill_between(pc_message_time_minutes, 0, pc_message_rates, 
                           where=(pc_message_rates < threshold_freq),
                           color='red', alpha=0.2, label='Below threshold')
        
        axes[1].set_ylabel('Message Rate (Hz)')
        axes[1].set_xlabel('Time (minutes)')
        title_prefix = f'{mouse_id} - ' if mouse_id else ''
        axes[1].set_title(f'{title_prefix}PC Scales Message Reception Rate - '
                        f'Total messages: {len(pc_timestamps)}, '
                        f'Missing: {pc_result.get("missing_messages", 0)}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim(0, max(35, np.max(pc_message_rates) * 1.1))
        
        # Add statistics box
        stats_text = (f'Min: {np.min(pc_message_rates):.1f} Hz\n'
                     f'Max: {np.max(pc_message_rates):.1f} Hz\n'
                     f'Std: {np.std(pc_message_rates):.1f} Hz')
        axes[1].text(0.02, 0.95, stats_text,
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add warning if rate is consistently low
        if median_rate < threshold_freq:
            axes[1].text(0.5, 0.5, f'⚠️ WARNING: Message rate below {threshold_freq} Hz!', 
                        transform=axes[1].transAxes, ha='center', va='center',
                        fontsize=16, color='red', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        axes[1].text(0.5, 0.5, 'Insufficient PC data for plotting', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # Message timing visualisation - use larger bins for long recordings
    if pc_result['success'] and len(pc_result['timestamps']) > 0:
        # Show message density over time
        pc_time_minutes = pc_result['timestamps'] / 60
        recording_duration_minutes = pc_time_minutes[-1]
        
        # Adjust bin width based on recording duration
        if recording_duration_minutes > 30:
            bin_width = 1.0  # 1 minute bins for long recordings
            expected_count = 30 * 60  # messages per minute
            axes[2].set_ylabel('Messages per minute')
        else:
            bin_width = 0.1  # 6-second bins for short recordings
            expected_count = 30 * 6  # messages per 6 seconds
            axes[2].set_ylabel('Messages per 6s')
        
        # Create histogram bins
        bins = np.arange(0, pc_time_minutes[-1] + bin_width, bin_width)
        
        # Limit number of bins for performance
        if len(bins) > 500:
            bins = np.linspace(0, pc_time_minutes[-1], 500)
            bin_width = bins[1] - bins[0]
            expected_count = 30 * bin_width * 60
        
        counts, _ = np.histogram(pc_time_minutes, bins=bins)
        bin_centres = (bins[:-1] + bins[1:]) / 2
        
        axes[2].bar(bin_centres, counts, width=bin_width*0.8, alpha=0.7, color='red')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_title('PC Message Density')
        axes[2].grid(True, alpha=0.3)
        
        # Add expected count line
        axes[2].axhline(y=expected_count, color='green', linestyle='--', 
                       label=f'Expected (~{expected_count:.0f} messages/bin)')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No message timing data available', 
                    ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    
    print("Rendering plot...")
    
    # Save the plot if required
    if save:
        output_dir = "temp_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"scales_frequency_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Reduced DPI for faster save
        print(f"Plot saved to {output_path}")
    
    # Show the plot if required
    if show:
        if web_display:
            # Get local IP address
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            port = 8988  # Default WebAgg port
            
            print("\n" + "="*50)
            print("Web server started!")
            print(f"Access the plot at: http://{local_ip}:{port}")
            print(f"Or locally at: http://localhost:{port}")
            print("Press Ctrl+C in the terminal to stop the server")
            print("="*50 + "\n")
        plt.show()
    else:
        plt.close()


def get_session_files_from_cohort(cohort_dir, session_id, use_existing_cohort_info=True, OEAB_legacy=False):
    """
    Get file paths for a session using Cohort_folder.
    
    Parameters
    ----------
    cohort_dir : str or Path
        Path to cohort directory.
    session_id : str
        Session ID to retrieve (e.g., "250527_133843_mtao108-3e").
    use_existing_cohort_info : bool, optional
        Whether to use existing cohort info if available (default is True).
    OEAB_legacy : bool, optional
        Whether to use legacy OEAB folder structure (default is False).
    
    Returns
    -------
    dict
        Dictionary containing paths to h5_path and json_path, or None if session not found.
    """
    # Create cohort object
    cohort = Cohort_folder(
        cohort_dir,
        multi=True,
        portable_data=False,
        OEAB_legacy=OEAB_legacy,
        use_existing_cohort_info=use_existing_cohort_info,
        plot=False
    )
    
    # Get session info
    session_info = cohort.get_session(session_id)
    
    if session_info is None:
        print(f"Session {session_id} not found in cohort")
        return None
    
    # Extract file paths
    h5_path = session_info.get("raw_data", {}).get("arduino_DAQ_h5", None)
    json_path = session_info.get("raw_data", {}).get("behaviour_data", None)
    
    if h5_path in [None, "None"] or json_path in [None, "None"]:
        print(f"Missing files for session {session_id}")
        print(f"  H5 path: {h5_path}")
        print(f"  JSON path: {json_path}")
        return None
    
    return {
        "h5_path": h5_path,
        "json_path": json_path,
        "mouse_id": session_info.get("mouse_id", session_id.split('_')[2])  # Extract from session ID if not available
    }


def main():
    """Main function to analyse scales frequency from both DAQ and PC data sources using Cohort_folder."""
    # Configuration parameters
    show_plot = True
    save_plot = False
    web_display = True  # Set to True to display in web browser
    
    # Cohort directory and session ID
    cohort_dir = "/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment"
    session_id = "250528_140739_mtao101-3g"  # Example session ID
    
    session_data = [
    ("250530_150321_mtao108-3e", 0.795),
    ("250611_125320_mtao108-3e", -3.062),
    ("250624_143350_mtao101-3c", -5.648),
    ("250602_113554_mtao107-2a", -13.231),
    ("250609_124914_mtao108-3e", 21.422),
    ("250522_114207_mtao106-3a", 23.150),   # poor alignment
    ("250610_134958_mtao108-3e", 24.914),
    ("250529_131006_mtao106-3a", -26.693),
    ("250603_131545_mtao108-3e", -42.585),
    ("250522_130749_mtao107-2a", -343.102),
    ("250604_134248_mtao107-2a", -642.681),
    ("250604_153227_mtao106-3a", -665.036),
    ("250610_122940_mtao106-3a", -731.668), # poor alignment
    ("250602_100403_mtao106-3a", 768.291),  # poor alignment
    ("250603_111907_mtao106-3a", -1251.353),    # poor alignment
    ("250528_123545_mtao108-3e", 1828.919), # questionable
    ("250609_112131_mtao106-3a", 1917.676), # poor alignment
    ("250603_120358_mtao106-3a", -9487.507),
    ("250611_112601_mtao106-3a", -21302.817),
    ("250528_111659_mtao106-3a", 26451.698) # near very bad
]


    # 250527_121911_mtao106-3a
# 250527_133843_mtao108-3e
# 250527_150817_mtao101-3g
# 250603_110543_mtao106-3a 
# 250522_163010_mtao101-3b # good quality for some reason
# 250528_140739_mtao101-3g

    # Alternative session IDs to test:
    # session_id = "250527_121911_mtao106-3a"
    
    # Get file paths from cohort
    print("Retrieving session files from cohort...")
    session_files = get_session_files_from_cohort(
        cohort_dir, 
        session_id, 
        use_existing_cohort_info=True,
        OEAB_legacy=False
    )
    
    if session_files is None:
        print("Failed to retrieve session files. Exiting.")
        return
    
    h5_path = session_files["h5_path"]
    json_path = session_files["json_path"]
    mouse_id = session_files["mouse_id"]
    
    print(f"Found session files:")
    print(f"  H5 path: {h5_path}")
    print(f"  JSON path: {json_path}")
    print(f"  Mouse ID: {mouse_id}")
    
    # Analysis parameters
    scales_channel = "SCALES"
    daq_threshold = 0.5  # Threshold for DAQ data (typically voltage)
    pc_threshold = 20.0  # Threshold for PC data (weight in grams)
    frequency_threshold = 10.0  # Hz
    window_size = 10  # pulses
    
    print("\n" + "="*60)
    print("SCALES FREQUENCY ANALYSIS - DAQ vs PC COMPARISON")
    print("="*60)
    
    # Analyse DAQ data
    print("\n1. Analysing DAQ (H5) data...")
    print(f"   File: {h5_path}")
    print("-" * 60)
    
    daq_result = analyse_scales_pulses(
        h5_path, 
        scales_channel=scales_channel, 
        threshold=daq_threshold,
        verbose=True
    )
    
    # Analyse PC behaviour data
    print("\n\n2. Analysing PC behaviour (JSON) data...")
    print(f"   File: {json_path}")
    print("-" * 60)
    
    pc_result = analyse_scales_pulses_from_behaviour_json(
        json_path,
        threshold=pc_threshold,
        verbose=True
    )
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if daq_result['success'] and pc_result['success']:
        print(f"\nDAQ pulse frequency: {daq_result['pulse_rate_hz']:.2f} Hz")
        print(f"DAQ total pulses: {daq_result['total_pulses']}")
        
        print(f"\nPC message reception rate: {pc_result['pc_message_rate_hz']:.1f} Hz")
        print(f"PC total messages: {len(pc_result['timestamps'])}")
        print(f"PC missing messages: {pc_result.get('missing_messages', 0)}")
        
        print(f"\nRecording duration:")
        print(f"  DAQ:  {daq_result['recording_duration']:.1f} seconds")
        print(f"  PC:   {pc_result['recording_duration']:.1f} seconds")
        
        print(f"\nData acquisition rate:")
        print(f"  DAQ sampling: ~{len(daq_result['timestamps'])/daq_result['recording_duration']:.0f} Hz")
        print(f"  PC message rate: {pc_result['pc_message_rate_hz']:.1f} Hz")
        
        if pc_result.get('missing_messages', 0) > 0:
            print(f"\n⚠️  WARNING: PC data has {pc_result['missing_messages']} missing messages!")
        
        # Check thresholds
        if daq_result['pulse_rate_hz'] < frequency_threshold:
            print(f"\n⚠️  WARNING: DAQ pulse rate below {frequency_threshold} Hz threshold!")
        if pc_result['pc_message_rate_hz'] < frequency_threshold:
            print(f"\n⚠️  WARNING: PC message rate below {frequency_threshold} Hz threshold!")
            print("     This indicates a problem with the scales hardware!")
        elif pc_result['pc_message_rate_hz'] < 25:
            print(f"\n⚠️  WARNING: PC message rate below expected 30 Hz!")

        # Perform stability analysis
        print_stability_analysis(daq_result, pc_result)
        
        # Create comparison plot
        print("\nCreating comparison plot...")
        plot_frequency_over_time_comparison(
            daq_result,
            pc_result,
            mouse_id=mouse_id,
            window_size=50,
            threshold_freq=frequency_threshold,
            save=save_plot,
            show=show_plot,
            web_display=web_display,
            max_plot_points=1000
        )
        
    else:
        print("\n✗ Analysis failed for one or both data sources")
        if not daq_result['success']:
            print(f"  DAQ error: {daq_result['error']}")
        if not pc_result['success']:
            print(f"  PC error: {pc_result['error']}")


if __name__ == "__main__":
    main()

# # Too few matched sessions:
# 250527_121911_mtao106-3a
# 250527_133843_mtao108-3e
# 250527_150817_mtao101-3g
# 250603_110543_mtao106-3a
# 250522_163010_mtao101-3b
# 250528_140739_mtao101-3g


# session_data = [
#     ("250530_150321_mtao108-3e", 0.795),
#     ("250611_125320_mtao108-3e", -3.062),
#     ("250624_143350_mtao101-3c", -5.648),
#     ("250602_113554_mtao107-2a", -13.231),
#     ("250609_124914_mtao108-3e", 21.422),
#     ("250522_114207_mtao106-3a", 23.150),
#     ("250610_134958_mtao108-3e", 24.914),
#     ("250529_131006_mtao106-3a", -26.693),
#     ("250603_131545_mtao108-3e", -42.585),
#     ("250522_130749_mtao107-2a", -343.102),
#     ("250604_134248_mtao107-2a", -642.681),
#     ("250604_153227_mtao106-3a", -665.036),
#     ("250610_122940_mtao106-3a", -731.668),
#     ("250602_100403_mtao106-3a", 768.291),
#     ("250603_111907_mtao106-3a", -1251.353),
#     ("250528_123545_mtao108-3e", 1828.919),
#     ("250609_112131_mtao106-3a", 1917.676),
#     ("250603_120358_mtao106-3a", -9487.507),
#     ("250611_112601_mtao106-3a", -21302.817),
#     ("250528_111659_mtao106-3a", 26451.698)
# ]
