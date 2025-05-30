#!/usr/bin/env python3
# timestamp_analysis.py - Contains analysis functions for ephys timestamps
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
import os
import matplotlib.pyplot as plt

def analyze_timestamps(events: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Analyze timestamp data to identify timing variations, missed pulses, and drift.
    
    Args:
        events (List[Tuple[float, float]]): List of (high_timestamp, duration) tuples
    
    Returns:
        Dict[str, float]: Dictionary of analysis results
    """
    # If no events or just one event, return empty analysis
    if len(events) <= 1:
        return {}
    
    # Extract timestamps and durations
    timestamps = np.array([event[0] for event in events])
    durations = np.array([event[1] for event in events])
    
    # Calculate time differences between consecutive pulses
    time_diffs = np.diff(timestamps)
    
    # Basic statistics
    mean_diff = np.mean(time_diffs)
    median_diff = np.median(time_diffs)
    std_diff = np.std(time_diffs)
    min_diff = np.min(time_diffs)
    max_diff = np.max(time_diffs)
    
    # Mean duration and std
    mean_duration = np.mean(durations)
    duration_std = np.std(durations)
    
    # Missed pulse detection (intervals > 1.9x median)
    missed_pulse_threshold = 1.9 * median_diff
    missed_pulse_indices = np.where(time_diffs > missed_pulse_threshold)[0]
    missed_pulse_count = len(missed_pulse_indices)
    
    # Calculate expected timestamps based on median interval
    expected_timestamps = np.zeros_like(timestamps)
    expected_timestamps[0] = timestamps[0]  # Start at the same point
    for i in range(1, len(timestamps)):
        expected_timestamps[i] = expected_timestamps[0] + i * median_diff
    
    # Calculate timing jitter (difference between actual and expected)
    jitter = timestamps - expected_timestamps
    jitter_mean = np.mean(jitter)
    jitter_std = np.std(jitter)
    jitter_max = np.max(np.abs(jitter))
    
    # Long-term drift analysis
    total_drift = jitter[-1]
    recording_duration = timestamps[-1] - timestamps[0]
    drift_per_second = total_drift / recording_duration if recording_duration > 0 else 0
    
    # Linear regression to measure drift trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, jitter)
    
    # Compile analysis results
    analysis = {
        "pulse_count": len(timestamps),
        "recording_duration_s": recording_duration,
        "mean_interval_ms": mean_diff * 1000,
        "median_interval_ms": median_diff * 1000,
        "min_interval_ms": min_diff * 1000,
        "max_interval_ms": max_diff * 1000,
        "interval_std_ms": std_diff * 1000,
        "mean_duration_ms": mean_duration * 1000,
        "duration_std_ms": duration_std * 1000,
        "missed_pulse_threshold_ms": missed_pulse_threshold * 1000,
        "missed_pulse_count": missed_pulse_count,
        "missed_pulse_percent": (missed_pulse_count / len(timestamps)) * 100 if len(timestamps) > 0 else 0,
        "jitter_mean_ms": jitter_mean * 1000,
        "jitter_std_ms": jitter_std * 1000,
        "jitter_max_ms": jitter_max * 1000,
        "total_drift_ms": total_drift * 1000,
        "drift_rate_ms_per_s": drift_per_second * 1000,
        "drift_slope_ms_per_s": slope * 1000,
        "drift_r_squared": r_value**2,
        "drift_p_value": p_value
    }
    
    return analysis

def save_analysis_plots(data: Dict[int, List[Tuple[float, float]]], analysis: Dict[int, Dict[str, float]], 
                       output_folder: str, session_id: str, target_pin: Optional[int] = None) -> None:
    """
    Generate and save analysis plots for the timestamp data.
    
    Args:
        data (Dict[int, List[Tuple[float, float]]]): Dictionary mapping pin numbers to event lists
        analysis (Dict[int, Dict[str, float]]): Analysis results for each pin
        output_folder (str): Folder to save the plots
        session_id (str): Session ID for naming the plots
        target_pin (Optional[int]): Specific pin to create plots for, or None for all pins
    """
    # Determine which pins to plot
    pins_to_plot = [target_pin] if target_pin is not None else list(data.keys())
    
    for pin in pins_to_plot:
        if pin not in data or not data[pin]:
            continue
            
        events = data[pin]
        timestamps = np.array([event[0] for event in events])
        durations = np.array([event[1] for event in events])
        
        # Skip if too few events to analyze
        if len(timestamps) <= 1:
            continue
        
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs)
        missed_pulse_threshold = 1.9 * median_diff
        missed_pulse_indices = np.where(time_diffs > missed_pulse_threshold)[0]
        
        # Calculate expected timestamps and jitter
        expected_timestamps = np.zeros_like(timestamps)
        expected_timestamps[0] = timestamps[0]
        for i in range(1, len(timestamps)):
            expected_timestamps[i] = expected_timestamps[0] + i * median_diff
        
        jitter = timestamps - expected_timestamps
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_folder, 'analysis_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Figure 1: Pulse Intervals
        plt.figure(figsize=(12, 6))
        plt.hist(time_diffs*1000, bins=100, alpha=0.7)
        plt.axvline(np.mean(time_diffs)*1000, color='r', linestyle='--', 
                   label=f'Mean: {np.mean(time_diffs)*1000:.4f} ms')
        plt.axvline(median_diff*1000, color='g', linestyle='--', 
                   label=f'Median: {median_diff*1000:.4f} ms')
        plt.axvline(missed_pulse_threshold*1000, color='black', linestyle='--', 
                   label=f'Missed Pulse Threshold: {missed_pulse_threshold*1000:.4f} ms')
        plt.title(f'Histogram of Pulse Intervals (Pin {pin})')
        plt.xlabel('Interval (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{session_id}_pin_{pin}_intervals.png'))
        plt.close()
        
        # Figure 2: Pulse intervals over time with missed pulses highlighted
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps[1:], time_diffs*1000, 'b-', alpha=0.5)
        plt.axhline(median_diff*1000, color='g', linestyle='--', 
                   label=f'Median: {median_diff*1000:.4f} ms')
        plt.axhline(missed_pulse_threshold*1000, color='r', linestyle='--', 
                   label=f'Threshold: {missed_pulse_threshold*1000:.4f} ms')
        
        # Highlight missed pulses
        if len(missed_pulse_indices) > 0:
            plt.scatter(timestamps[missed_pulse_indices+1], 
                       time_diffs[missed_pulse_indices]*1000, 
                       color='red', s=30, label='Potential Missed Pulses')
        
        plt.title(f'Pulse Intervals Over Time (Pin {pin})')
        plt.xlabel('Time (s)')
        plt.ylabel('Interval (ms)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{session_id}_pin_{pin}_intervals_over_time.png'))
        plt.close()
        
        # Figure 3: Jitter analysis
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, jitter*1000, 'b-', alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f'Timing Jitter (Pin {pin})')
        plt.xlabel('Time (s)')
        plt.ylabel('Jitter (ms)')
        plt.savefig(os.path.join(plots_dir, f'{session_id}_pin_{pin}_jitter.png'))
        plt.close()
        
        # Figure 4: Drift analysis with linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, jitter)
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, jitter*1000, 'b-', alpha=0.5, label='Actual drift')
        plt.plot(timestamps, (intercept + slope*timestamps)*1000, 'r--', 
                 label=f'Linear fit (slope: {slope*1000:.8f} ms/s, R²: {r_value**2:.4f})')
        plt.title(f'Cumulative Timing Drift (Pin {pin})')
        plt.xlabel('Time (s)')
        plt.ylabel('Drift (ms)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{session_id}_pin_{pin}_drift.png'))
        plt.close()
        
        # Figure 5: Pulse durations
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, durations*1000, 'g-', alpha=0.5)
        plt.axhline(np.median(durations)*1000, color='r', linestyle='--', 
                   label=f'Median Duration: {np.median(durations)*1000:.4f} ms')
        plt.title(f'Pulse Durations Over Time (Pin {pin})')
        plt.xlabel('Time (s)')
        plt.ylabel('Duration (ms)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{session_id}_pin_{pin}_durations.png'))
        plt.close()