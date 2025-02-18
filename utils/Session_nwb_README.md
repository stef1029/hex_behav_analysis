# Behavioral Trial Analysis Tool

## Overview
This Python script provides a comprehensive toolkit for analyzing behavioral trial data from experimental sessions. It's designed to process and analyze data from experimental setups involving:
- Video recordings
- LED cues
- Sensor interactions
- DeepLabCut (DLC) tracking data
- NWB (Neurodata Without Borders) file format

## Key Features
- Trial detection and analysis
- Video frame extraction and processing
- DeepLabCut coordinate processing
- Behavioral angle calculations
- NWB file handling
- LED visualization
- Multi-phase experiment support

## Main Components

### Session Class
The primary class that handles individual experimental sessions. Key functionalities include:
- Loading and processing session data
- Managing video data
- Processing DLC coordinates
- Calculating behavioral angles
- Adding video data to trials
- LED visualization

### DetectTrials Class
Handles the detection and processing of experimental trials. Features include:
- Trial detection based on different experimental phases
- Processing of LED and sensor data
- Trial merging for complex experimental protocols
- Go-cue detection and validation

## Dependencies
- pathlib
- json
- pandas
- bisect
- cv2 (OpenCV)
- math
- numpy
- datetime
- pynwb
- DeepLabCut

## Data Structure Requirements

### NWB File Structure
The NWB files should contain:
- Acquisition data (scales, sensors)
- Stimulus data (LEDs, valves)
- Video data
- DLC coordinates (optional)
- Session metadata

### Session Directory Structure
Expected directory structure:
```
session_directory/
├── NWB_file
├── DLC_coords.csv (optional)
├── behavior_video
└── other_session_files
```

## Usage

### Basic Usage
```python
from session_analysis import Session

# Create a session object
session = Session(session_dict)

# Access processed trials
trials = session.trials

# Visualize LED positions
session.draw_LEDs(start=0, end=None, output_path="output_directory")
```

### Session Dictionary Format
```python
session_dict = {
    'directory': 'path/to/session',
    'session_id': 'YYMMDD_HHMMSS',
    'portable': False,
    'processed_data': {
        'NWB_file': 'path/to/nwb',
        'DLC': {
            'coords_csv': 'path/to/dlc_coords.csv'
        }
    }
}
```

## Trial Data Structure
Each trial contains:
- Start and end times
- Correct port information
- Sensor interaction data
- Video frame indices
- DLC tracking data
- Behavioral angle data
- Success/failure status

## Features

### Video Processing
- Frame extraction from specific trial periods
- LED visualization
- Integration with DLC tracking data

### Angle Calculations
- Mouse heading calculations
- Port angle calibration
- Relative angle computations

### Data Integration
- Synchronization of video frames with trial data
- Integration of DLC coordinates
- Sensor interaction mapping

## Error Handling
The script includes comprehensive error handling for:
- File loading failures
- Video processing errors
- Data synchronization issues
- Missing or corrupted data

## Output
- Processed trial data
- Annotated videos
- Behavioral metrics
- Angular measurements

## Object Structure

### Session Object Structure
```
Session
├── Attributes
│   ├── session_dict            # Original session dictionary
│   ├── session_directory       # Path to session directory
│   ├── session_ID             # Session identifier
│   ├── portable               # Boolean flag for portable setup
│   ├── nwb_file_path         # Path to NWB file
│   ├── DLC_coords_path       # Path to DLC coordinates CSV
│   ├── last_timestamp        # Last timestamp in session
│   ├── phase                 # Experimental phase
│   ├── rig_id               # Experimental rig identifier
│   ├── session_video        # Path to behavior video
│   ├── video_timestamps     # Array of video frame timestamps
│   ├── trials              # List of trial dictionaries
│   ├── port_angles         # List of calibrated port angles
│   └── port_coordinates    # List of port coordinate tuples
│
├── Methods
│   ├── load_data()                    # Loads session data from NWB file
│   ├── add_DLC_coords_to_nwb()        # Adds DLC coordinates to NWB file
│   ├── frametime_to_index()           # Converts frame times to indices
│   ├── add_video_data_to_trials()     # Adds video data to trial objects
│   ├── add_angle_data()               # Adds angle calculations to trials
│   ├── find_angles()                  # Calculates angles for a trial
│   ├── draw_LEDs()                    # Visualizes LED positions
│   ├── find_file()                    # Utility for finding files
│   ├── find_dir()                     # Utility for finding directories
│   ├── is_number()                    # Utility for number validation
│   └── calibrate_port_angles()        # Calibrates port angle positions
```

### Trial Dictionary Structure
```
Trial Dictionary
├── Basic Trial Info
│   ├── trial_no               # Trial number in sequence
│   ├── phase                  # Experimental phase
│   ├── correct_port           # Target port number (1-6 or 'audio-1')
│   ├── cue_start             # Timestamp of cue start
│   └── cue_end               # Timestamp of cue end
│
├── Sensor Data
│   ├── sensor_touches        # List of all sensor interactions
│   │   └── [For each touch]
│   │       ├── sensor_touched    # Sensor ID
│   │       ├── sensor_start      # Touch start time
│   │       └── sensor_end        # Touch end time
│   │
│   ├── next_sensor           # First sensor interaction
│   │   ├── sensor_touched    # Sensor ID
│   │   ├── sensor_start      # Touch start time
│   │   └── sensor_end        # Touch end time
│   │
│   └── success              # Boolean for correct port touch
│
├── Video Data
│   ├── video_frames         # List of relevant frame indices
│   └── DLC_data            # DataFrame of DLC tracking data
│       ├── timestamps      # Frame timestamps
│       └── [For each body part]
│           ├── x           # X coordinate
│           ├── y           # Y coordinate
│           └── likelihood  # Confidence score
│
├── Angle Data
│   └── turn_data
│       ├── bearing              # Mouse heading angle
│       ├── port_position        # Target port coordinates
│       ├── midpoint            # Mouse position coordinates
│       ├── cue_presentation_angle  # Angle to target from heading
│       └── port_touched_angle      # Angle to touched port
│
└── Phase-Specific Data
    ├── catch                # Boolean for catch trials (Phase 10)
    └── go_cue               # Go cue timestamp (Phase 9c)
```

## Notes
- The script is designed for specific experimental setups with 6 ports
- Phase-specific processing is implemented
- Supports multiple experimental protocols
- Handles both audio and visual cues

## Contributing
When contributing to this project:
1. Document any new features or modifications
2. Maintain consistent error handling
3. Update tests for new functionality
4. Follow the existing code structure

## Limitations
- Specific to certain experimental setups
- Requires specific NWB file structure
- Video processing can be memory-intensive
- Dependent on correct DLC tracking data

## Troubleshooting
Common issues and solutions:
- Video frame misalignment: Check frame indexing
- DLC data missing: Verify CSV file format
- NWB file errors: Confirm file structure
- Angle calculation issues: Verify port calibration