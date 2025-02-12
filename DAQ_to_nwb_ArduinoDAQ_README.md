# DAQ to NWB Conversion Tool

## Overview
This Python script converts Data Acquisition (DAQ) experimental data into the Neurodata Without Borders (NWB) format. It handles various types of experimental data including stimulus events, behavioral measurements, video recordings, and scales data.

## Key Features
- Conversion of DAQ data to NWB format
- Time series synchronization
- Interval detection and processing
- Video data integration
- Scales data processing
- Comprehensive metadata handling

## Dependencies
- pynwb
- numpy
- h5py
- datetime
- dateutil
- pathlib
- uuid

## Functions

### Core Functions Structure
```
DAQ to NWB Tool
├── Time Series Processing
│   ├── timeseries_to_intervals()
│   │   ├── Input: timestamps, signal
│   │   ├── Parameters:
│   │   │   ├── HIGH: signal high value
│   │   │   └── filter: duration filtering flag
│   │   └── Output: intervals, timestamps
│   │
│   └── intervals_to_digital_events()
│       ├── Input: intervals, interval_timestamps
│       └── Output: digital events, timestamps
│
└── Main Conversion
    └── DAQ_to_nwb()
        ├── Input Parameters
        │   ├── DAQ_h5_path
        │   ├── scales_data
        │   ├── session_ID
        │   ├── mouse_id
        │   ├── video_directory
        │   ├── video_timestamps
        │   ├── session_directory
        │   ├── session_metadata
        │   ├── session_description
        │   ├── experimenter
        │   ├── institution
        │   └── lab
        └── Output: NWBFile
```

### Data Structures

#### NWB File Structure
```
NWBFile
├── Metadata
│   ├── session_description
│   ├── identifier (UUID)
│   ├── session_start_time
│   ├── experimenter
│   ├── institution
│   ├── lab
│   └── experiment_description
│       ├── phase
│       ├── rig
│       ├── wait
│       └── cue
│
├── Subject Info
│   ├── subject_id
│   ├── species
│   └── weight
│
├── Stimulus Data
│   ├── Buzzer Events (1-6)
│   │   ├── timestamps
│   │   └── intervals
│   ├── LED Events (1-6)
│   │   ├── timestamps
│   │   └── intervals
│   ├── Cue Events
│   │   ├── GO_CUE
│   │   └── NOGO_CUE
│   ├── Spotlight Data (1-6)
│   │   ├── brightness values
│   │   └── timestamps
│   └── Valve Events (1-6)
│       ├── duration
│       └── timestamps
│
├── Acquisition Data
│   ├── Sensor Events (1-6)
│   │   ├── timestamps
│   │   └── intervals
│   ├── Scales Data
│   │   ├── weights
│   │   ├── timestamps
│   │   └── threshold
│   └── Video Data
│       ├── file reference
│       └── timestamps
```

## Usage

### Basic Usage
```python
from daq_to_nwb import DAQ_to_nwb

# Convert DAQ data to NWB
nwbfile = DAQ_to_nwb(
    DAQ_h5_path="path/to/daq.h5",
    scales_data=scales_dict,
    session_ID="YYYYMMDD_HHMMSS",
    mouse_id="mouse_id",
    video_directory=video_path,
    video_timestamps=timestamps_dict,
    session_directory=session_path,
    session_metadata=metadata_dict,
    session_description="Experiment description",
    experimenter="Name",
    institution="Institution",
    lab="Lab name"
)
```

### Input Requirements

#### DAQ HDF5 File Structure
```
daq.h5
├── timestamps         # Array of timestamps
└── channel_data      # Group containing channel datasets
    ├── BUZZER1-6     # Buzzer event data
    ├── LED_1-6       # LED event data
    ├── SPOT1-6       # Spotlight intensity data
    ├── VALVE1-6      # Valve event data
    └── SENSOR1-6     # Sensor event data
```

#### Scales Data Dictionary
```python
scales_data = {
    'weights': [],           # Array of weight measurements
    'timestamps': [],        # Array of measurement timestamps
    'mouse_weight_threshold': float  # Weight threshold value
}
```

#### Session Metadata Dictionary
```python
session_metadata = {
    'behaviour_phase': str,  # Experimental phase
    'rig_id': int,          # Rig identifier
    'wait_duration': float, # Wait period duration
    'cue_duration': float,  # Cue presentation duration
    'mouse_weight': float   # Mouse weight
}
```

## Features

### Time Series Processing
- Signal interval detection
- Duration filtering
- Digital event conversion
- Timestamp synchronization

### Data Integration
- Multiple data stream handling
- Video synchronization
- Scales data integration
- Metadata compilation

### Stimulus Processing
- Buzzer events
- LED events
- Cue presentations
- Valve rewards
- Spotlight intensity

### Behavioral Data
- Sensor interactions
- Weight measurements
- Video recordings

## Error Handling
- File existence verification
- Data integrity checks
- Timestamp validation
- Format validation

## Output
- NWB file containing:
  - Synchronized time series
  - Processed intervals
  - Integrated metadata
  - Referenced video data
  - Behavioral measurements

## Performance Considerations
- Efficient array operations
- Memory management for large datasets
- Timestamp precision handling
- File size optimization

## Limitations
- Specific file format requirements
- Fixed channel structure
- Memory constraints for large datasets
- Video file referencing requirements

## Troubleshooting
Common issues and solutions:
- Missing timestamps: Check DAQ file integrity
- Channel mismatch: Verify DAQ structure
- Video sync issues: Check timestamp alignment
- Scales data errors: Verify data format

## Notes
- Compatible with multiple experimental phases
- Supports various stimulus types
- Handles multiple sensor channels
- Integrates with NWB ecosystem