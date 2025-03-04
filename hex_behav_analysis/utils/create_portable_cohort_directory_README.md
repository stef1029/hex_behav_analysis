# Create portable cohort directories:

## Overview
This Python script provides functionality for processing and synchronizing NWB (Neurodata Without Borders) files between directories, adding behavior coordinates when missing, and maintaining data organization across storage locations.

## Key Features
- Automated NWB file processing
- Behavioral coordinates validation and addition
- File synchronization using rsync
- Progress tracking with color coding
- Cohort information preservation

## Dependencies
- pathlib
- subprocess
- colorama
- pynwb
- Custom modules:
  - Cohort_folder
  - Session_nwb

## Function Structure

### Main Components
```
NWB Sync Tool
├── Initialization
│   └── create_cohort_folder()
│       ├── Input: source directory
│       └── Output: Cohort object
│
├── File Processing
│   └── check_and_process_nwb_file()
│       ├── Input
│       │   ├── nwb_file_path
│       │   └── cohort
│       └── Operations
│           ├── Check for behaviour_coords
│           └── Add coordinates if missing
│
└── File Synchronization
    └── process_and_rsync_files()
        ├── Input
        │   ├── source directory
        │   ├── destination directory
        │   └── cohort
        └── Operations
            ├── File identification
            ├── NWB processing
            ├── Directory structure preservation
            └── File synchronization
```

## Usage

### Setting Up Directories
Edit the path variables in the `main()` function to specify your source and destination directories:

```python
def main():
    # Edit these paths for your use case
    source_directory = Path(r'/path/to/source/data')
    destination_directory = Path(r'/path/to/destination/data')

    # Initialize cohort
    cohort = create_cohort_folder(source_directory)

    # Execute the processing and rsync process
    process_and_rsync_files(source_directory, destination_directory, cohort)
```

You can uncomment and modify the appropriate paths in the code. Examples of common setups are provided in the `main()` function:

```python
# Example 1: July WT cohort
# source_directory = Path(r'/cephfs2/srogers/Behaviour code/2407_July_WT_cohort/Data')
# destination_directory = Path(r'/cephfs2/srogers/Behaviour code/2407_July_WT_cohort/Portable_data')

# Example 2: March training
# source_directory = Path(r'/cephfs2/srogers/March_training')
# destination_directory = Path(r'/cephfs2/srogers/Behaviour code/March_training_portable')
```

### Running the Script
After setting your paths, simply run the script:
```bash
python nwb_sync.py
```

### File Requirements
The script processes:
- `.nwb` files (NWB format files)
- `cohort_info.png` (Cohort information visualization)
- Excludes files in 'OEAB' directories

## Features

### Cohort Management
- Creates and manages cohort folder structure
- Maintains cohort metadata
- Preserves directory hierarchy

### NWB File Processing
- Validates behavior coordinates presence
- Automatically adds missing coordinates
- Preserves existing data integrity

### File Synchronization
- Uses rsync for efficient file transfer
- Maintains file structure
- Progress tracking with visual feedback

### Progress Monitoring
- Color-coded progress indicators
- File-by-file status updates
- Error reporting

## Directory Structure
```
Working Directory
├── Source Directory
│   ├── .nwb files
│   ├── cohort_info.png
│   └── subdirectories
│
└── Destination Directory
    ├── Synchronized .nwb files
    ├── cohort_info.png
    └── Maintained structure
```

## Error Handling
- File existence verification
- Directory creation checks
- NWB file validation
- Process monitoring

## Output
- Processed NWB files with behavior coordinates
- Synchronized directory structure
- Progress reports
- Transfer status updates

## Performance Considerations
- Efficient file transfer using rsync
- Selective file processing
- Directory structure preservation
- Memory-efficient operations

## Limitations
- Requires rsync availability
- Specific file format requirements
- Memory constraints for large files
- Network dependency for transfers

## Troubleshooting
Common issues and solutions:
- Missing files: Check source paths
- Sync errors: Verify rsync installation
- Processing failures: Check NWB file integrity
- Directory access: Verify permissions

## Notes
- Supports multiple cohort structures
- Handles large file transfers
- Preserves data organization
- Provides visual progress feedback