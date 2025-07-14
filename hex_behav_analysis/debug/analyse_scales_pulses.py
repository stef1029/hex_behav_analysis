"""
Script to analyse scales pulse frequencies across a cohort and identify sessions below threshold.
Only analyses sessions conducted on rig 3.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from tqdm import tqdm


def check_rig_number(behaviour_data_path):
    """
    Check if a session was conducted on a specific rig by reading the behaviour data JSON.
    
    Parameters
    ----------
    behaviour_data_path : str
        Path to the behaviour data JSON file.
    
    Returns
    -------
    int or None
        Rig number if found, None otherwise.
    """
    try:
        with open(behaviour_data_path, 'r') as f:
            behaviour_data = json.load(f)
            return behaviour_data.get('Rig', None)
    except Exception:
        return None


def analyse_scales_pulses(h5_path, scales_channel="SCALES", threshold=0.5):
    """
    Analyse scales pulses from Arduino DAQ data.
    
    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing Arduino DAQ data.
    scales_channel : str, optional
        Name of the scales channel in the HDF5 file (default is "SCALES").
    threshold : float, optional
        Threshold value for pulse detection (default is 0.5).
    
    Returns
    -------
    dict
        Dictionary containing:
        - success: bool indicating if analysis was successful
        - pulse_rate_hz: float representing pulse rate in Hz (None if failed)
        - total_pulses: int representing total number of pulses detected
        - error: str containing error message if analysis failed
    """
    try:
        print(f"{h5_path}")
        with h5py.File(h5_path, 'r') as h5f:
            timestamps = h5f['timestamps'][:]
            
            if scales_channel not in h5f['channel_data']:
                return {'success': False, 'error': 'Channel not found'}
            
            scales_signal = h5f['channel_data'][scales_channel][:]
        
        # Detect pulses by finding rising edges
        signal = np.asarray(scales_signal)
        above_thresh = signal >= threshold
        rising = (above_thresh[1:] == True) & (above_thresh[:-1] == False)
        edges = np.where(rising)[0] + 1
        
        if len(edges) < 2:
            return {
                'success': True,
                'pulse_rate_hz': 0,
                'total_pulses': len(edges)
            }
        
        # Calculate pulse rate from median interval
        pulse_times = timestamps[edges]
        intervals = np.diff(pulse_times)
        median_interval = np.median(intervals)
        pulse_rate = 1.0 / median_interval if median_interval > 0 else 0
        
        return {
            'success': True,
            'pulse_rate_hz': pulse_rate,
            'total_pulses': len(edges)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def check_cohort_scales_frequencies(cohort_obj, frequency_threshold=10.0, target_rig=3):
    """
    Check scales pulse frequencies across entire cohort for sessions on a specific rig.
    
    Parameters
    ----------
    cohort_obj : Cohort_folder
        Cohort object containing session information.
    frequency_threshold : float, optional
        Minimum acceptable frequency in Hz (default is 10.0).
    target_rig : int, optional
        Rig number to filter sessions by (default is 3).
        
    Returns
    -------
    list of tuple
        List containing tuples of (mouse_id, session_id, frequency) for sessions 
        below threshold on the target rig.
    """
    low_frequency_sessions = []
    total_sessions = 0
    rig_3_sessions = 0
    skipped_wrong_rig = 0
    skipped_no_behaviour_data = 0
    
    # Count total sessions first
    for mouse_id, mouse_data in cohort_obj.cohort["mice"].items():
        for session_id, session_info in mouse_data["sessions"].items():
            h5_path = session_info.get("raw_data", {}).get("arduino_DAQ_h5", None)
            if h5_path not in [None, "None"]:
                total_sessions += 1
    
    print(f"Total sessions found: {total_sessions}")
    print(f"Filtering for rig {target_rig} sessions only...")
    
    # Analyse sessions with progress bar
    with tqdm(total=total_sessions, desc="Analysing scales frequencies") as pbar:
        for mouse_id, mouse_data in cohort_obj.cohort["mice"].items():
            for session_id, session_info in mouse_data["sessions"].items():
                h5_path = session_info.get("raw_data", {}).get("arduino_DAQ_h5", None)
                
                if h5_path in [None, "None"]:
                    continue
                
                # Update progress bar description
                pbar.set_description(f"Analysing {mouse_id}/{session_id}")
                
                # Check rig number from behaviour data
                behaviour_data_path = session_info.get("raw_data", {}).get("behaviour_data", None)
                if behaviour_data_path in [None, "None"]:
                    skipped_no_behaviour_data += 1
                    pbar.update(1)
                    continue
                
                rig_number = check_rig_number(behaviour_data_path)
                if rig_number != target_rig:
                    skipped_wrong_rig += 1
                    pbar.update(1)
                    continue
                
                rig_3_sessions += 1
                
                # Analyse session
                result = analyse_scales_pulses(h5_path)
                
                if result['success'] and result['pulse_rate_hz'] < frequency_threshold:
                    low_frequency_sessions.append((
                        mouse_id,
                        session_id,
                        result['pulse_rate_hz']
                    ))
                
                pbar.update(1)
    
    print(f"\nSummary:")
    print(f"  - Sessions on rig {target_rig}: {rig_3_sessions}")
    print(f"  - Sessions on other rigs: {skipped_wrong_rig}")
    print(f"  - Sessions without behaviour data: {skipped_no_behaviour_data}")
    
    return low_frequency_sessions


def main():
    """Main function to check scales frequencies across cohort for rig 3 sessions."""
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    
    # Initialise cohort
    cohort_dir = "/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment"
    cohort = Cohort_folder(cohort_dir, OEAB_legacy=False)
    frequency_threshold = 10.0  # Hz
    target_rig = 3
    
    # Check frequencies for rig 3 sessions only
    low_freq_sessions = check_cohort_scales_frequencies(
        cohort, 
        frequency_threshold=frequency_threshold,
        target_rig=target_rig
    )
    
    # Print results
    print(f"\n\nRig {target_rig} sessions with scales frequency < {frequency_threshold} Hz:")
    print("=" * 60)
    
    if low_freq_sessions:
        print("\nlow_frequency_sessions = [")
        for mouse_id, session_id, freq in low_freq_sessions:
            print(f"    ('{mouse_id}', '{session_id}', {freq:.2f}),  # {freq:.2f} Hz")
        print("]")
        print(f"\nTotal: {len(low_freq_sessions)} sessions below threshold on rig {target_rig}")
    else:
        print(f"No sessions found with frequency below threshold on rig {target_rig}")


if __name__ == "__main__":
    main()