# Raw Behavior Data Processing Tool

## Overview
This Python script provides a comprehensive toolkit for processing raw behavioral experimental data, converting various data formats into standardized NWB files, and managing multi-session experiments. It handles video processing, data synchronization, and experimental metadata management.

## Key Features
- Raw data processing and validation
- Video frame synchronization
- Scales data processing
- Multi-processing support for video conversion
- NWB file generation
- Automated batch processing
- Comprehensive error logging

## Dependencies
- pathlib
- os
- matplotlib
- json
- cv2 (OpenCV)
- datetime
- multiprocessing
- subprocess
- struct
- h5py
- logging
- numpy
- pynwb

## Object Structure

### Process_Raw_Behaviour_Data Object Structure
```
Process_Raw_Behaviour_Data
├── Attributes
│   ├── session                 # Session information dictionary
│   ├── session_id             # Unique session identifier
│   ├── mouse_id               # Mouse identifier
│   ├── data_folder_path       # Path to data directory
│   ├── raw_video_Path         # Path to raw video file
│   ├── behaviour_data_Path    # Path to behavior data file
│   ├── tracker_data_Path      # Path to tracking data file
│   ├── arduino_DAQ_Path       # Path to Arduino DAQ file
│   ├── rig_id                 # Experimental rig identifier
│   ├── video_fps              # Video frames per second
│   ├── timestamps             # Array of timestamps
│   ├── camera_pulses          # Camera synchronization pulses
│   ├── frame_times            # Dictionary of frame timestamps
│   ├── scales_data            # Dictionary of scales data
│   └── video_metadata         # Video metadata dictionary
│
├── Methods
│   ├── Data Processing
│   │   ├── ingest_behaviour_data()    # Main processing pipeline
│   │   ├── clean_DAQ_data()          # Cleans DAQ data
│   │   ├── pulse_ID_sync()           # Synchronizes pulse IDs
│   │   ├── get_camera_frame_times()   # Extracts frame timestamps
│   │   └── get_scales_data()         # Processes scales data
│   │
│   ├── Video Processing
│   │   ├── bmp_to_avi_MP()           # Converts BMP to AVI
│   │   ├── get_dims()                # Gets video dimensions
│   │   ├── concatenate_videos()       # Combines video chunks
│   │   └── clear_BMP_files()         # Cleanup utility
│   │
│   └── Utilities
│       ├── count_files()             # File counting utility
│       ├── find_file()               # File location utility
│       └── find_dir()                # Directory location utility
```

### Data Structures

#### Session Dictionary Structure
```
Session Dictionary
├── directory           # Path to session directory
├── session_id          # Unique session identifier
├── mouse_id           # Mouse identifier
├── portable           # Portable data flag
│
├── raw_data
│   ├── raw_video      # Video file path
│   ├── behaviour_data # Behavior data file path
│   ├── tracker_data   # Tracking data file path
│   ├── arduino_DAQ_h5 # DAQ data file path
│   └── is_all_raw_data_present?  # Data completeness flag
│
└── processed_data
    ├── sendkey_logs         # Processed log file path
    ├── video_frametimes     # Frame timing data
    ├── sendkey_metadata     # Processed metadata
    ├── NWB_file            # NWB file path
    └── preliminary_analysis_done?  # Analysis status flag
```

#### Scales Data Structure
```
Scales Data Dictionary
├── timestamps           # Array of measurement timestamps
├── weights             # Array of weight measurements
├── pulse_IDs          # Array of pulse identifiers
├── sendkey_timestamps  # Array of sendkey timestamps
├── mouse_weight_threshold  # Weight threshold value
└── scales_type        # Type of scales (wired/wireless)
```

## Usage

### Basic Usage
```python
from process_raw_behaviour import Process_Raw_Behaviour_Data

# Process a single session
processor = Process_Raw_Behaviour_Data(session_info, logger)

# Batch processing with multiprocessing
def main_MP():
    cohort_directory = Path("path/to/cohort")
    Cohort = Cohort_folder(cohort_directory, multi=True)
    
    # Process multiple sessions
    for session in sessions_to_process:
        Process_Raw_Behaviour_Data(session, logger)
```

## Data Requirements

### Required Files
```
session_directory/
├── raw.avi                 # Raw video recording
├── behaviour_data.json     # Behavioral event data
├── Tracker_data.json      # Tracking metadata
├── ArduinoDAQ.h5         # DAQ recordings
└── temp/                 # Temporary processing directory
```

## Features

### Data Synchronization
- Camera frame synchronization
- DAQ pulse synchronization
- Scales data integration
- Video frame timing extraction

### Video Processing
- BMP to AVI conversion
- Multi-process video handling
- Frame dropping detection
- Video metadata extraction

### Data Validation
- File completeness checks
- Data integrity validation
- Frame count verification
- Timestamp alignment verification

### Output Generation
- NWB file creation
- Processed data logs
- Synchronized timestamps
- Analysis status reports

## Error Handling
- Comprehensive error logging
- Data validation checks
- Frame dropping detection
- File presence verification
- Timestamp synchronization validation

## Output Files
- Processed NWB files
- Video frame timing files
- Sendkey log files
- Error logs
- Analysis status reports

## Performance Considerations
- Multi-processing support for video conversion
- Memory-efficient data handling
- Optimized file operations
- Scalable batch processing

## Limitations
- Specific file format requirements
- Memory intensive for large videos
- Requires specific hardware setup
- Limited to certain experimental protocols

## Troubleshooting
Common issues and solutions:
- Missing timestamps: Check DAQ synchronization
- Frame dropping: Verify camera setup
- Data synchronization: Check pulse alignment
- File format errors: Verify input data structure

## Notes
- Supports both wired and wireless scales
- Handles multiple experimental phases
- Compatible with various hardware setups
- Integrates with NWB data standard