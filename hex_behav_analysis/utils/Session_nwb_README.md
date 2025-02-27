# Session Analysis Script Documentation

## Overview
This script is designed to analyze behavioral experiments with mice, where the mice interact with a platform that has 6 ports arranged in a circle. The script processes video recordings of these experiments, tracking the mouse's movements and interactions with the ports.

## What Does This Script Do?

The script's main purpose is to:
1. Load experimental session data
2. Process video recordings
3. Track mouse movements using DeepLabCut (DLC)
4. Detect and analyze trials
5. Calculate angles between the mouse and ports
6. Save all this information for later use

## Getting Started

### Creating a Session Dictionary
The first step is to create a session dictionary, which is required to create a Session object. The easiest way to do this is:

1. Create a Cohort_folder object by passing your cohort directory:
   ```python
   cohort = Cohort_folder("/path/to/your/cohort/folder")
   ```

2. Use the cohort object to get your session dictionary:
   ```python
   session_dict = cohort.get_session("your_session_id")
   ```

### Creating a New Session

When you create a new Session object (e.g., `session = Session(session_dict)`), the following happens:

1. **Initial Setup**
   - Loads basic session information (ID, directory paths)
   - Finds necessary files (video recordings, data files)
   - Checks if analysis has been done before

2. **Data Loading**
   - Loads experimental data from NWB files (special scientific data format)
   - Gets important metadata like:
     - Which experimental phase this is
     - Which experimental rig was used
     - Timestamps for the entire session

3. **Trial Detection**
   - Identifies individual trials in the session
   - Each trial represents one interaction sequence where:
     - A cue is presented
     - The mouse responds
     - A reward might be given

4. **Video Analysis**
   - For each trial, the script:
     - Identifies relevant video frames
     - Gets tracking data (where the mouse's ears are)
     - Calculates the mouse's heading direction
     - Determines angles between the mouse and ports

### Data Storage

The script saves processed data in two ways:
1. **Analysis Pickle Files**: Quick-access files for later use
2. **NWB Files**: Standard scientific data format for sharing

## Important Terms

- **Trial**: One complete interaction sequence
- **DLC (DeepLabCut)**: Software that tracks mouse body parts in the video
- **NWB**: A standard file format for scientific data
- **Port**: One of six interaction points on the experimental platform
- **Cue**: A signal telling the mouse which port to approach

## Key Features

1. **Automatic Trial Detection**
   - Finds where trials start and end
   - Groups relevant data for each trial

2. **Mouse Tracking**
   - Tracks mouse ear positions
   - Calculates which way the mouse is facing
   - Determines angles between mouse and ports

3. **Video Processing**
   - Can draw helpful markers on videos
   - Shows which ports are active
   - Visualizes mouse movement data

4. **Data Saving**
   - Saves processed data for quick reloading
   - Prevents need to reanalyze everything each time

## Common Use Cases

1. **Analyzing New Sessions**
   ```python
   session = Session(session_dict)
   ```
   This loads and processes all the data for a new session.

2. **Reprocessing Data**
   ```python
   session = Session(session_dict, recalculate=True)
   ```
   This forces the script to reanalyze everything, even if it's been done before.

3. **Visualizing Results**
   ```python
   session.draw_LEDs(output_path="your_folder")
   ```
   This creates a video showing the analyzed session with helpful markers.

## Tips for Users

1. **First Time Setup**
   - Make sure all required files are in the correct directories
   - Check that video files and data files match

2. **Error Handling**
   - If something goes wrong, the script will print helpful error messages
   - Common issues include missing files or mismatched data

3. **Data Checking**
   - The script saves its work frequently
   - You can always rerun analysis if needed
   - Previous analysis is saved and won't be lost

## Technical Requirements

- Python with specific libraries installed
- Video files from the experiment
- NWB data files
- DeepLabCut tracking data

## Getting Help

If you encounter issues:
1. Check that all files are present
2. Look for error messages in the output
3. Make sure file paths are correct
4. Verify that video files are not corrupted

Remember: This script is designed to be robust and will try to tell you what's wrong if something fails.