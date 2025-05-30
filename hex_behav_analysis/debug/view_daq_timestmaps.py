import h5py
import numpy as np
from pathlib import Path

def print_hdf5_timestamps(file_path, num_timestamps=100):
    """
    Print the first N timestamps from an ArduinoDAQ HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
        num_timestamps (int): Number of timestamps to print
    """
    try:
        with h5py.File(file_path, 'r') as h5f:
            # Check if timestamps dataset exists
            if 'timestamps' not in h5f:
                print(f"Error: No timestamps dataset found in {file_path}")
                return False
                
            # Get timestamps dataset
            timestamps = h5f['timestamps']
            total_timestamps = len(timestamps)
            
            # How many to print
            to_print = min(num_timestamps, total_timestamps)
            
            # Print file info
            print(f"\n{'='*80}")
            print(f"File: {file_path}")
            print(f"Total timestamps: {total_timestamps}")
            
            # Print HDF5 attributes
            print("\nFile Attributes:")
            for attr_name, attr_value in h5f.attrs.items():
                print(f"  {attr_name}: {attr_value}")
                
            # If "timestamp_source" and "messages_per_second" are present, highlight them
            if 'timestamp_source' in h5f.attrs:
                print(f"\n* Timestamp source: {h5f.attrs['timestamp_source']}")
            if 'messages_per_second' in h5f.attrs:
                print(f"* Sample rate: {h5f.attrs['messages_per_second']} Hz")
            
            # Print timestamps
            print(f"\nFirst {to_print} timestamps:")
            print("  [index] value (diff from previous)")
            
            for i in range(to_print):
                if i == 0:
                    diff = 0
                else:
                    diff = timestamps[i] - timestamps[i-1]
                    
                print(f"  [{i:5d}] {timestamps[i]:.6f} (+{diff:.6f})")
                
            # Show some statistics about the timestamps
            if total_timestamps > 1:
                diffs = np.diff(timestamps[:min(1000, total_timestamps)])
                avg_diff = np.mean(diffs)
                min_diff = np.min(diffs)
                max_diff = np.max(diffs)
                
                print(f"\nTimestamp Statistics (first 1000 intervals):")
                print(f"  Average interval: {avg_diff:.6f} seconds")
                print(f"  Minimum interval: {min_diff:.6f} seconds")
                print(f"  Maximum interval: {max_diff:.6f} seconds")
                print(f"  Estimated sample rate: {1.0/avg_diff:.2f} Hz")
                
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    # ===== CONFIGURATION - EDIT THIS PATH =====
    
    # Direct path to the HDF5 file you want to examine
    file_path = r"Z:\Behaviour code\2409_September_cohort\DATA_ArduinoDAQ\241017_151131\241017_151132_mtao89-1e\241017_151132_mtao89-1e-ArduinoDAQ.h5"
    
    # Number of timestamps to print
    num_timestamps = 100
    
    # ===== END OF CONFIGURATION =====
    
    # Process the file
    print_hdf5_timestamps(file_path, num_timestamps)
    
    print("\nDone!")

if __name__ == "__main__":
    main()