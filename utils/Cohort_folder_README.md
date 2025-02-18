# Cohort Data Management Tool

## Overview
This Python script provides a comprehensive toolkit for managing experimental cohort data, organizing individual mouse sessions, and handling both raw and processed data. The tool is designed to automatically organize experimental data files, track analysis status, and provide cohort-level visualizations.

## Key Features
- Automated mouse session organization
- Raw and processed data validation
- Data visualization and reporting
- NWB file integration
- Support for both single and multi-session experiments
- Portable and legacy data handling

## Main Components

### Cohort_folder Class
The primary class that manages cohort-level data organization and analysis tracking. Features include:
- Session detection and organization
- Data completeness validation
- Analysis status tracking
- Data visualization
- Metadata extraction and management

## Dependencies
- json
- pathlib
- cv2 (OpenCV)
- pandas
- matplotlib
- seaborn
- numpy
- re
- pynwb
- datetime

## Data Structure Requirements

### Directory Structure
```
cohort_directory/
├── mouse_sessions/
│   ├── YYYYMMDD_HHMMSS_mouseID/
│   │   ├── raw_data/
│   │   ├── processed_data/
│   │   └── analysis_output/
│   └── ...
├── cohort_info.json
├── concise_cohort_info.json
└── cohort_info.png
```

### Required Files
- Behavior data files
- Video recordings
- Tracking data
- DAQ data (Arduino)
- OEAB recordings (optional)
- DLC files (optional)

## Object Structure

### Cohort_folder Object Structure
```
Cohort_folder
├── Attributes
│   ├── cohort_directory      # Path to main cohort directory
│   ├── multi                 # Boolean for multi-session setup
│   ├── plot                  # Boolean for plot generation
│   ├── portable_data        # Boolean for portable data mode
│   ├── OEAB_legacy         # Boolean for legacy OEAB support
│   ├── session_folders     # List of session directory paths
│   ├── json_filename       # Path to cohort info JSON
│   └── cohort              # Main data dictionary
│
├── Methods
│   ├── Data Initialization
│   │   ├── init_portable_data()     # Initialize portable data structure
│   │   ├── init_raw_data()          # Initialize raw data structure
│   │   └── find_mice()              # Detect and organize mouse sessions
│   │
│   ├── Data Access
│   │   ├── get_session()            # Retrieve specific session data
│   │   ├── phases()                 # Get phase-specific session data
│   │   └── find_nwbs()             # Locate NWB files
│   │
│   ├── Data Validation
│   │   ├── check_raw_data()         # Validate raw data completeness
│   │   ├── check_for_preliminary_analysis()  # Check analysis status
│   │   └── find_DLC_files()         # Locate DLC-related files
│   │
│   ├── Visualization
│   │   ├── graphical_cohort_info()  # Generate cohort visualizations
│   │   └── text_summary_cohort_info()  # Generate text summaries
│   │
│   └── Utilities
│       ├── find_file()              # File location utility
│       ├── find_OEAB_dir()          # OEAB directory location
│       └── get_session_metadata()    # Extract session metadata
```

### Cohort Dictionary Structure
```
Cohort Dictionary
├── Cohort Info
│   ├── Cohort name           # Name of the cohort
│   └── mice                  # Dictionary of mouse data
│
├── Mouse Data
│   └── [Mouse ID]
│       └── sessions          # Dictionary of session data
│
├── Session Data
│   └── [Session ID]
│       ├── directory         # Session directory path
│       ├── mouse_id          # Mouse identifier
│       ├── session_id        # Session identifier
│       ├── portable          # Portable data flag
│       │
│       ├── raw_data
│       │   ├── raw_video            # Video file path
│       │   ├── behaviour_data       # Behavior data file path
│       │   ├── tracker_data         # Tracking data file path
│       │   ├── arduino_DAQ_json     # DAQ data file path
│       │   ├── OEAB                 # OEAB directory path
│       │   ├── scales_data          # Scales data status
│       │   ├── video_length         # Video duration
│       │   ├── session_metadata     # Session metadata
│       │   ├── is_all_raw_data_present?  # Data completeness flag
│       │   └── missing_files        # List of missing files
│       │
│       └── processed_data
│           ├── sendkey_logs         # Processed log file path
│           ├── video_frametimes     # Frame timing data
│           ├── sendkey_metadata     # Processed metadata
│           ├── NWB_file            # NWB file path
│           ├── DLC                  # DLC-related files
│           └── preliminary_analysis_done?  # Analysis status flag
```

## Usage

### Basic Usage
```python
from cohort_folder import Cohort_folder

# Initialize cohort folder
cohort = Cohort_folder(
    cohort_directory="path/to/cohort",
    multi=True,
    plot=True,
    portable_data=False,
    OEAB_legacy=True
)

# Access session data
session_data = cohort.get_session("YYYYMMDD_HHMMSS")

# Get phase-specific data
phase_data = cohort.phases()

# Generate visualization
cohort.graphical_cohort_info()
```

## Features

### Data Organization
- Automatic session detection
- Mouse ID extraction
- File completeness validation
- Analysis status tracking

### Visualization and Reporting
- Heatmap generation
- Trial count visualization
- Phase progression tracking
- Text-based summaries

### Data Validation
- Raw data completeness checks
- Processed data validation
- Missing file detection
- File structure verification

## Error Handling
- Directory existence validation
- File presence verification
- Data integrity checks
- Metadata extraction validation

## Output Files
- cohort_info.json: Complete cohort data
- concise_cohort_info.json: Simplified cohort data
- cohort_info.png: Visual cohort summary
- cohort_summary.txt: Text-based summary

## Notes
- Supports both single and multi-session experiments
- Handles legacy OEAB data formats
- Compatible with portable data structures
- Integrates with NWB file format

## Contributing
When contributing to this project:
1. Maintain consistent file structure handling
2. Update visualization capabilities as needed
3. Test with both legacy and current data formats
4. Document any new data structures or formats

## Limitations
- Specific file naming conventions required
- Directory structure must follow specified format
- Some features require specific file formats
- Visualization may be memory-intensive for large cohorts

## Troubleshooting
Common issues and solutions:
- Missing files: Check file naming conventions
- Data validation failures: Verify file formats
- Visualization errors: Check data completeness
- OEAB detection issues: Verify directory structure