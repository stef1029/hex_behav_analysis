Session Dictionary Structure
===========================

Root Level Keys:
├── directory (str)
├── mouse_id (str)
├── portable (bool)
├── session_id (str)
├── processed_data (dict)
│   ├── DLC (dict)
│   │   ├── coords_csv (str) - Path to CSV coordinates file
│   │   ├── coords_h5 (str) - Path to H5 coordinates file
│   │   ├── labelled_video (str) - Path to labelled video (or "None")
│   │   ├── meta_pickle (str) - Path to metadata pickle file
│   │   └── model_name (str) - DLC model identifier
│   ├── NWB_file (str) - Path to Neurodata Without Borders file
│   ├── preliminary_analysis_done? (bool)
│   ├── sendkey_logs (str) - Path to sendkey logs CSV
│   ├── sendkey_metadata (str) - Path to sendkey metadata JSON
│   └── video_frametimes (str) - Path to video frame times JSON
└── raw_data (dict)
    ├── arduino_DAQ_h5 (str) - Path to Arduino DAQ H5 file
    ├── behaviour_data (str) - Path to behaviour data JSON
    ├── is_all_raw_data_present? (bool)
    ├── missing_files (list)
    ├── raw_video (str) - Path to raw video AVI file
    ├── scales_data (bool)
    ├── session_metadata (dict)
    │   ├── cue_duration (str) - Duration in ms
    │   ├── phase (str) - Experimental phase identifier
    │   ├── total_trials (int) - Number of trials
    │   └── wait_duration (str) - Wait duration in ms
    ├── tracker_data (str) - Path to tracker data JSON
    └── video_length (int) - Video length in minutes
Quick Access Reference:
python# Basic session info
session['directory']
session['mouse_id']
session['portable']
session['session_id']

# Processed data paths
session['processed_data']['DLC']['coords_csv']
session['processed_data']['DLC']['coords_h5']
session['processed_data']['DLC']['labelled_video']
session['processed_data']['DLC']['meta_pickle']
session['processed_data']['DLC']['model_name']
session['processed_data']['NWB_file']
session['processed_data']['preliminary_analysis_done?']
session['processed_data']['sendkey_logs']
session['processed_data']['sendkey_metadata']
session['processed_data']['video_frametimes']

# Raw data paths and info
session['raw_data']['arduino_DAQ_h5']
session['raw_data']['behaviour_data']
session['raw_data']['is_all_raw_data_present?']
session['raw_data']['missing_files']
session['raw_data']['raw_video']
session['raw_data']['scales_data']
session['raw_data']['tracker_data']
session['raw_data']['video_length']

# Session metadata
session['raw_data']['session_metadata']['cue_duration']
session['raw_data']['session_metadata']['phase']
session['raw_data']['session_metadata']['total_trials']
session['raw_data']['session_metadata']['wait_duration']