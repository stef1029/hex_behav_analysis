from hex_behav_analysis.utils.Session_nwb import Session
from pathlib import Path
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from pynwb import NWBHDF5IO
import numpy as np

def inspect_session_data(session_dict):
    """
    Inspect the data structures in the NWB file to diagnose dimension mismatches.
    """
    print("\n===== DETAILED SESSION DATA INSPECTION =====")
    
    # Get the NWB file path
    if session_dict.get('portable', False):
        nwb_file_path = Path(session_dict.get('NWB_file'))
    else:
        nwb_file_path = Path(session_dict.get('processed_data', {}).get('NWB_file'))
    
    print(f"Examining NWB file: {nwb_file_path}")
    
    # Open the NWB file for inspection
    with NWBHDF5IO(str(nwb_file_path), 'r') as io:
        nwbfile = io.read()
        
        # Check if 'behaviour_video' exists in acquisition
        if 'behaviour_video' in nwbfile.acquisition:
            video_data = nwbfile.acquisition['behaviour_video']
            video_timestamps = video_data.timestamps[:]
            print(f"Video timestamps length: {len(video_timestamps)}")
            print(f"Video timestamps range: {video_timestamps[0]} to {video_timestamps[-1]}")
        else:
            print("No 'behaviour_video' found in acquisition")
        
        # Check if 'behaviour_coords' exists in processing
        if 'behaviour_coords' in nwbfile.processing:
            behaviour_module = nwbfile.processing['behaviour_coords']
            
            print("\nBody part data in 'behaviour_coords':")
            for name in behaviour_module.data_interfaces:
                ts = behaviour_module.data_interfaces[name]
                ts_data = ts.data[:]
                ts_timestamps = ts.timestamps[:]
                
                print(f"\nBody part: {name}")
                print(f"  Data shape: {ts_data.shape}")
                print(f"  Timestamps length: {len(ts_timestamps)}")
                
                if len(ts_data.shape) > 1:
                    print(f"  Data first dimension: {ts_data.shape[0]}")
                    print(f"  Data second dimension: {ts_data.shape[1]}")
                
                # Check for dimension mismatch
                if ts_data.shape[0] != len(ts_timestamps):
                    print(f"  !! MISMATCH: Data first dimension ({ts_data.shape[0]}) != Timestamps length ({len(ts_timestamps)})")
                    
                    # Check if transposing would fix the issue
                    if len(ts_data.shape) > 1 and ts_data.shape[1] == len(ts_timestamps):
                        print(f"  !! SOLUTION: Transposing would fix the mismatch (second dimension {ts_data.shape[1]} matches timestamps)")
        else:
            print("No 'behaviour_coords' found in processing")
        
        # Check for TimeSeries in acquisition
        print("\nChecking TimeSeries in acquisition:")
        for name, timeseries in nwbfile.acquisition.items():
            if hasattr(timeseries, 'timestamps') and hasattr(timeseries, 'data'):
                try:
                    data = timeseries.data[:]
                    timestamps = timeseries.timestamps[:]
                    
                    print(f"\nTimeSeries: {name}")
                    print(f"  Type: {type(timeseries).__name__}")
                    print(f"  Data shape: {data.shape}")
                    print(f"  Timestamps length: {len(timestamps)}")
                    
                    # Check for dimension mismatch
                    if isinstance(data, np.ndarray) and len(data.shape) > 0:
                        if data.shape[0] != len(timestamps):
                            print(f"  !! MISMATCH: Data first dimension ({data.shape[0]}) != Timestamps length ({len(timestamps)})")
                except Exception as e:
                    print(f"  Error accessing data/timestamps: {str(e)}")
        
        # Check for TimeSeries in stimulus
        print("\nChecking TimeSeries in stimulus:")
        for name, timeseries in nwbfile.stimulus.items():
            if hasattr(timeseries, 'timestamps') and hasattr(timeseries, 'data'):
                try:
                    data = timeseries.data[:]
                    timestamps = timeseries.timestamps[:]
                    
                    print(f"\nTimeSeries: {name}")
                    print(f"  Type: {type(timeseries).__name__}")
                    print(f"  Data shape: {data.shape}")
                    print(f"  Timestamps length: {len(timestamps)}")
                    
                    # Check for dimension mismatch
                    if isinstance(data, np.ndarray) and len(data.shape) > 0:
                        if data.shape[0] != len(timestamps):
                            print(f"  !! MISMATCH: Data first dimension ({data.shape[0]}) != Timestamps length ({len(timestamps)})")
                except Exception as e:
                    print(f"  Error accessing data/timestamps: {str(e)}")
        
        # Check processing modules
        print("\nChecking processing modules:")
        for module_name in nwbfile.processing:
            module = nwbfile.processing[module_name]
            print(f"Processing module: {module_name}")
            
            for interface_name in module.data_interfaces:
                interface = module.data_interfaces[interface_name]
                print(f"  Interface: {interface_name} ({type(interface).__name__})")
                
                # If it's a TimeSeries
                if hasattr(interface, 'timestamps') and hasattr(interface, 'data'):
                    try:
                        data = interface.data[:]
                        timestamps = interface.timestamps[:]
                        
                        print(f"    Data shape: {data.shape}")
                        print(f"    Timestamps length: {len(timestamps)}")
                        
                        # Check for dimension mismatch
                        if isinstance(data, np.ndarray) and len(data.shape) > 0:
                            if data.shape[0] != len(timestamps):
                                print(f"    !! MISMATCH: Data first dimension ({data.shape[0]}) != Timestamps length ({len(timestamps)})")
                    except Exception as e:
                        print(f"    Error accessing data/timestamps: {str(e)}")

# Main script
test_cohort = Path(r"/cephfs2/dwelch/Behaviour/test_2")
cohort = Cohort_folder(test_cohort, multi=True, OEAB_legacy=False)
session = cohort.get_session("250211_110245_wtjp280-4a")
print(f"Session dictionary: {session}")

# Run inspection before creating Session object
inspect_session_data(session)

# Create Session object with additional debugging
try:
    print("\n===== CREATING SESSION OBJECT =====")
    test_Session = Session(session)
    print("Session object created successfully")
except Exception as e:
    print(f"Error creating session object: {str(e)}")
    import traceback
    traceback.print_exc()