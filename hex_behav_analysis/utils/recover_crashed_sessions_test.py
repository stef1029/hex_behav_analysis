import os
import csv
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

def log_info(message, verbose=True):
    """Print info message in blue"""
    if verbose:
        print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")

def log_success(message, verbose=True):
    """Print success message in green"""
    if verbose:
        print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

def log_warning(message, verbose=True):
    """Print warning message in yellow"""
    if verbose:
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

def log_error(message, verbose=True):
    """Print error message in red"""
    if verbose:
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")

def log_debug(message, verbose=False):
    """Print debug message in cyan if verbose mode is enabled"""
    if verbose:
        print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")

def is_already_processed(file_path, operation_type):
    """
    Checks if a file has already been processed by looking for a marker file.
    
    Args:
        file_path (Path): Path to the file being processed
        operation_type (str): Type of operation ("frame_id_recovery" or "hdf5_conversion")
    
    Returns:
        bool: True if already processed, False otherwise
    """
    marker_file = Path(f"{file_path}_{operation_type}_processed")
    return marker_file.exists()

def mark_as_processed(file_path, operation_type, verbose=False):
    """
    Creates a marker file to indicate that a file has been processed.
    
    Args:
        file_path (Path): Path to the file being processed
        operation_type (str): Type of operation ("frame_id_recovery" or "hdf5_conversion") 
        verbose (bool): Whether to print debug information
    """
    marker_file = Path(f"{file_path}_{operation_type}_processed")
    with open(marker_file, 'w') as f:
        f.write(f"Processed on {datetime.now()}")
    log_debug(f"Created marker file: {marker_file}", verbose)

def check_hdf5_integrity(hdf5_path, verbose=False):
    """
    Checks if an HDF5 file exists and can be opened without errors.
    
    Args:
        hdf5_path (Path): Path to the HDF5 file
        verbose (bool): Whether to print error information
    
    Returns:
        bool: True if file exists and can be read, False otherwise
    """
    if not hdf5_path.exists():
        return False
        
    try:
        with h5py.File(hdf5_path, 'r') as h5f:
            # Try to read some data to verify integrity
            keys = list(h5f.keys())
            if 'channel_data' in keys:
                channel_keys = list(h5f['channel_data'].keys())
                if len(channel_keys) > 0:
                    # Try reading a small sample of data
                    sample = h5f['channel_data'][channel_keys[0]][0:10]
            return True
    except Exception as e:
        log_error(f"HDF5 file {hdf5_path} exists but is corrupted: {str(e)}", verbose)
        return False

def check_json_integrity(json_path, verbose=False):
    """
    Checks if a JSON file exists and can be loaded without errors.
    
    Args:
        json_path (Path): Path to the JSON file
        verbose (bool): Whether to print error information
    
    Returns:
        tuple: (bool, dict) - (True if file exists and can be read, loaded data or None)
    """
    if not json_path.exists():
        return False, None
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return True, data
    except Exception as e:
        log_error(f"JSON file {json_path} exists but is corrupted: {str(e)}", verbose)
        return False, None

def recover_frame_ids(json_file_path, verbose=False):
    """
    Recovers frame IDs from backup file if the JSON file has empty frame_IDs or is corrupted.
    Also ensures that required fields for downstream processing are present.
    
    Args:
        json_file_path (str or Path): Path to the JSON file
        verbose (bool): Whether to print detailed information
    
    Returns:
        bool: True if recovery was successful, False otherwise
    """
    try:
        json_path = Path(json_file_path)
        
        # Check if already processed
        if is_already_processed(json_path, "frame_id_recovery"):
            log_debug(f"File already processed: {json_path.name}, skipping...", verbose)
            return False
            
        # Check JSON file integrity and load data if possible
        is_valid, data = check_json_integrity(json_path, verbose)
        if not is_valid:
            log_warning(f"JSON file is corrupted or not found: {json_file_path}", verbose)
        elif data.get('frame_IDs') and len(data['frame_IDs']) > 0:
            # Check for required fields
            required_fields = ['start_time', 'end_time', 'frame_rate']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                log_debug(f"Frame IDs already exist in {json_path.name} and all required fields are present", verbose)
                mark_as_processed(json_path, "frame_id_recovery", verbose)
                return False
            else:
                log_warning(f"JSON file {json_path.name} is missing required fields: {missing_fields}", verbose)
                # Continue processing to add the missing fields
        
        # Construct backup file path based on JSON filename pattern
        # Extract the base name from JSON file (remove '_Tracker_data.json')
        json_stem = json_path.stem
        if json_stem.endswith('_Tracker_data'):
            base_name = json_stem[:-13]  # Remove '_Tracker_data' suffix
        else:
            base_name = json_stem
            
        # Construct backup filename: base_name + '_frame_ids_backup.txt'
        backup_filename = f"{base_name}_frame_ids_backup.txt"
        backup_path = json_path.parent / backup_filename
        
        # Add more debug logging for backup file
        log_debug(f"Looking for backup file at: {backup_path.absolute()}", verbose)
        log_debug(f"Backup file exists: {backup_path.exists()}", verbose)
        
        if not backup_path.exists():
            log_error(f"Backup file not found: {backup_path}", verbose)
            return False
            
        log_info(f"\nProcessing {json_path.name}", verbose)
        log_info(f"Reading backup from: {backup_path.name}", verbose)
        
        # Read frame IDs from backup file
        frame_ids = []
        with open(backup_path, 'r') as f:
            for line in f:
                try:
                    frame_id = int(line.strip())
                    frame_ids.append(frame_id)
                except ValueError as e:
                    log_warning(f"Invalid frame ID in backup file: {line.strip()}", verbose)
                    continue
        
        if not frame_ids:
            log_error("No valid frame IDs found in backup file", verbose)
            return False
            
        log_info(f"Found {len(frame_ids)} frame IDs", verbose)
        
        # Create new data dictionary if the original was corrupted
        if not is_valid:
            data = {}
        
        # Add required fields if missing
        if 'start_time' not in data:
            data['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_debug(f"Adding missing 'start_time' field with value: {data['start_time']}", verbose)
            
        if 'frame_rate' not in data:
            data['frame_rate'] = 30.0  # Default FPS value
            log_debug(f"Adding missing 'frame_rate' field with default value: {data['frame_rate']}", verbose)
            
        if 'end_time' not in data or data['end_time'] == "":
            # Calculate end_time based on fps and number of frames
            fps = data['frame_rate']
            num_frames = len(frame_ids)
            
            if fps > 0 and num_frames > 0:
                # Calculate recording duration in seconds
                duration_seconds = num_frames / fps
                
                try:
                    # Try multiple date formats for start_time
                    start_dt = None
                    formats_to_try = [
                        "%Y-%m-%d %H:%M:%S",  # Standard format
                        "%y%m%d_%H%M%S",      # Format like "241017_151132"
                        "%y%m%d%H%M%S"        # Another possible format
                    ]
                    
                    for date_format in formats_to_try:
                        try:
                            start_dt = datetime.strptime(data['start_time'], date_format)
                            log_debug(f"Successfully parsed start_time with format: {date_format}", verbose)
                            break
                        except ValueError:
                            continue
                            
                    if start_dt:
                        # Add the duration to get the end time
                        # Calculate hours, minutes, seconds from total seconds
                        hours, remainder = divmod(duration_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        # Add time to start_dt
                        end_dt = start_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
                        data['end_time'] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                        log_debug(f"Calculated end_time based on {num_frames} frames at {fps} fps = {duration_seconds:.2f}s", verbose)
                    else:
                        # If all parsing attempts fail
                        raise ValueError(f"Could not parse start_time: {data['start_time']}")
                        
                except Exception as e:
                    # If parsing fails, just set end_time to duration from now
                    log_warning(f"Failed to parse start_time: {e}", verbose)
                    now = datetime.now()
                    hours, remainder = divmod(duration_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    end_dt = now + timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    data['end_time'] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Fallback to a default 5 minute duration if we can't calculate
                log_warning(f"Could not calculate accurate duration: fps={fps}, frames={num_frames}", verbose)
                try:
                    # Try multiple formats for start_time
                    start_dt = None
                    formats_to_try = [
                        "%Y-%m-%d %H:%M:%S",
                        "%y%m%d_%H%M%S",
                        "%y%m%d%H%M%S"
                    ]
                    
                    for date_format in formats_to_try:
                        try:
                            start_dt = datetime.strptime(data['start_time'], date_format)
                            break
                        except ValueError:
                            continue
                            
                    if start_dt:
                        end_dt = start_dt + timedelta(minutes=5)
                        data['end_time'] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        raise ValueError("Could not parse start_time")
                except Exception:
                    now = datetime.now()
                    data['end_time'] = (now + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            
            log_debug(f"Added end_time field with value: {data['end_time']}", verbose)
            
        # Update the frame IDs
        data['frame_IDs'] = frame_ids
        
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        log_success(f"Successfully updated {json_path.name} with {len(frame_ids)} frame IDs and required fields", verbose)
        
        # Mark as processed
        mark_as_processed(json_path, "frame_id_recovery", verbose)
        return True
        
    except Exception as e:
        log_error(f"Error processing {json_file_path}: {str(e)}", verbose)
        return False
    

def recover_from_backup(backup_file_path, verbose=False):
    """
    Recovers data from a backup CSV file and converts it to proper HDF5 format.
    Only processes the backup if no corresponding HDF5 file exists or if it's corrupted.
    
    Handles variable-length messages by truncating or padding to match expected channel count.
    
    If the CSV contains timestamps in a third column, it will use those timestamps instead
    of generating them based on sample rate.
    
    Args:
        backup_file_path (str or Path): Path to the backup CSV file
        verbose (bool): Whether to print detailed information
    
    Returns:
        bool: True if recovery was successful, False otherwise
    """
    backup_path = Path(backup_file_path)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file_path}")
        
    # Check if already processed
    if is_already_processed(backup_path, "hdf5_conversion"):
        log_debug(f"Backup already processed: {backup_path.name}, skipping...", verbose)
        return False
        
    # Check if HDF5 file already exists and is valid
    folder_name = backup_path.stem.replace('-backup', '')
    hdf5_path = backup_path.parent / f"{folder_name}-ArduinoDAQ.h5"
    
    if check_hdf5_integrity(hdf5_path, verbose):
        log_debug(f"Skipping {backup_path} - HDF5 file already exists and is valid", verbose)
        mark_as_processed(backup_path, "hdf5_conversion", verbose)
        return False
    
    # Extract folder name and mouse ID from the backup file name
    parts = folder_name.split('_')
    date_time = '_'.join(parts[:2])
    mouse_ID = '_'.join(parts[2:])
    
    # Read the backup CSV file
    messages_from_arduino = []
    actual_timestamps = []
    has_timestamps = False
    message_length_stats = {}  # Track different message lengths
    
    try:
        log_info(f"\nProcessing backup file: {backup_path}", verbose)
        log_info(f"Date time: {date_time}", verbose)
        log_info(f"Mouse ID: {mouse_ID}", verbose)
        
        # First, check if there's a third column by reading a few rows
        with open(backup_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            sample_rows = []
            for i, row in enumerate(csv_reader):
                if i < 5:  # Just read first 5 rows to check format
                    sample_rows.append(row)
                else:
                    break
            
            # Check if any row has 3 columns (assumes consistent format)
            has_timestamps = any(len(row) >= 3 for row in sample_rows)
            
            if has_timestamps:
                log_info(f"Detected timestamp data in the CSV file, will use actual timestamps", verbose)
            else:
                log_info(f"No timestamp data found, will generate timestamps based on sample rate", verbose)
        
        # Now read the full file
        with open(backup_path, 'r') as csvfile:
            # Debug first few lines if verbose
            if verbose:
                first_lines = []
                for i, line in enumerate(csvfile):
                    if i < 5:
                        first_lines.append(line.strip())
                log_debug(f"First few lines of data:", verbose)
                for line in first_lines:
                    log_debug(f"  {line}", verbose)
                
                # Reset file pointer to beginning
                csvfile.seek(0)
            
            csv_reader = csv.reader(csvfile)
            skipped_rows = 0
            invalid_timestamp_count = 0
            
            for row_num, row in enumerate(csv_reader):
                # Handle varying row formats
                if len(row) < 2:
                    log_warning(f"Skipping row {row_num} with insufficient columns: {row}", verbose)
                    skipped_rows += 1
                    continue
                    
                try:
                    message_id = int(row[0])
                    message_data = int(row[1])
                    
                    # Check message data bit length to track statistics
                    bit_length = message_data.bit_length()
                    message_length_stats[bit_length] = message_length_stats.get(bit_length, 0) + 1
                    
                    messages_from_arduino.append([message_id, message_data])
                    
                    # If there's a third column with timestamps, collect it
                    if len(row) >= 3 and row[2]:
                        try:
                            timestamp = float(row[2])
                            actual_timestamps.append(timestamp)
                        except ValueError:
                            log_warning(f"Invalid timestamp value in row {row_num}: {row[2]}", verbose)
                            invalid_timestamp_count += 1
                            # If we encounter invalid timestamp, revert to using sample rate
                            if len(actual_timestamps) > 0:
                                log_warning(f"Some timestamps were invalid, reverting to sample rate calculation", verbose)
                                actual_timestamps = []
                                has_timestamps = False
                            
                    elif has_timestamps:
                        # If we expected timestamps but this row doesn't have one
                        log_warning(f"Missing timestamp in row {row_num}: {row}", verbose)
                        # Revert to using sample rate
                        if len(actual_timestamps) > 0:
                            log_warning(f"Some rows missing timestamps, reverting to sample rate calculation", verbose)
                            actual_timestamps = []
                            has_timestamps = False
                    
                except ValueError as e:
                    log_warning(f"Error parsing row {row_num} {row}: {e}", verbose)
                    skipped_rows += 1
                    continue
            
            if skipped_rows > 0:
                log_warning(f"Skipped {skipped_rows} invalid rows during processing", verbose)
            
            if invalid_timestamp_count > 0:
                log_warning(f"Found {invalid_timestamp_count} invalid timestamps", verbose)
                    
        if not messages_from_arduino:
            raise ValueError("No valid data could be parsed from the CSV file")
        
        # Log message length statistics
        if verbose and message_length_stats:
            log_info("Message length distribution:", verbose)
            for bit_length, count in sorted(message_length_stats.items()):
                log_info(f"  {bit_length} bits: {count} messages ({count/len(messages_from_arduino)*100:.1f}%)", verbose)
            
        # Verify we have timestamps for all messages if using timestamp column
        if has_timestamps and len(actual_timestamps) != len(messages_from_arduino):
            log_warning(f"Mismatch in number of timestamps ({len(actual_timestamps)}) and messages ({len(messages_from_arduino)})", verbose)
            log_warning(f"Reverting to sample rate calculation", verbose)
            has_timestamps = False
            actual_timestamps = []
            
    except Exception as e:
        log_error(f"Error reading backup file: {str(e)}", verbose)
        return False
    
    # Channel definitions
    channel_indices = (
        "SPOT2", "SPOT3", "SPOT4", "SPOT5", "SPOT6", "SPOT1", "SENSOR6", "SENSOR1",
        "SENSOR5", "SENSOR2", "SENSOR4", "SENSOR3", "BUZZER4", "LED_3", "LED_4",
        "BUZZER3", "BUZZER5", "LED_2", "LED_5", "BUZZER2", "BUZZER6", "LED_1",
        "LED_6", "BUZZER1", "VALVE4", "VALVE3", "VALVE5", "VALVE2", "VALVE6",
        "VALVE1", "GO_CUE", "NOGO_CUE", "CAMERA", "SCALES"
    )
    
    num_channels = len(channel_indices)
    num_messages = len(messages_from_arduino)
    
    log_info(f"Processing {num_messages} messages with {num_channels} channels", verbose)
    
    # Process data with robust handling of variable message lengths
    message_ids = np.array([message[0] for message in messages_from_arduino], dtype=np.uint32)
    message_data = np.array([message[1] for message in messages_from_arduino], dtype=np.uint64)
    channel_data_array = np.zeros((num_messages, num_channels), dtype=np.uint8)
    
    # Process data with proper zero-padding for shorter messages and skipping of longer ones
    valid_messages = []
    valid_message_indices = []
    skipped_long_messages = 0
    
    for i, message in enumerate(message_data):
        # Check if message is too long (indicates corruption)
        if message.bit_length() > num_channels:
            skipped_long_messages += 1
            if skipped_long_messages <= 5:  # Only log first few occurrences
                log_debug(f"Message {i}: Skipped message with {message.bit_length()} bits (> {num_channels})", verbose)
            continue
        
        # Convert to binary representation with proper zero-padding
        # Always pad to num_channels width to handle leading zeros correctly
        binary_repr = np.binary_repr(message, width=num_channels)
        binary_array = np.array(list(binary_repr), dtype=np.uint8)
        binary_array = binary_array[::-1]  # Reverse for little-endian
        
        valid_messages.append(binary_array)
        valid_message_indices.append(i)
    
    # Create channel data array from valid messages only
    if not valid_messages:
        raise ValueError("No valid messages found after filtering")
        
    channel_data_array = np.array(valid_messages, dtype=np.uint8)
    
    # Update message arrays to only include valid messages
    message_ids = message_ids[valid_message_indices]
    message_data = message_data[valid_message_indices]
    num_messages = len(valid_messages)
    
    # Log summary of adjustments made
    if skipped_long_messages > 0:
        log_warning(f"Skipped {skipped_long_messages} messages that were longer than {num_channels} bits", verbose)
    
    log_info(f"Successfully processed {num_messages} valid messages (shorter messages were zero-padded)", verbose)
    
    # Generate timestamps
    sample_rate = 1000  # Default sample rate in Hz
    
    # Update timestamps to match valid messages only
    if has_timestamps and actual_timestamps:
        # Filter timestamps to match valid messages
        timestamps = np.array([actual_timestamps[i] for i in valid_message_indices])
        log_info(f"Using {len(timestamps)} actual timestamps from CSV data (filtered to valid messages)", verbose)
        
        # Calculate effective sample rate for metadata (average)
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            if time_span > 0:
                effective_rate = (len(timestamps) - 1) / time_span
                sample_rate = effective_rate
                log_info(f"Calculated effective sample rate: {effective_rate:.2f} Hz", verbose)
    else:
        # Generate timestamps based on sample rate
        timestamps = np.arange(num_messages) / sample_rate
        log_info(f"Generated {len(timestamps)} timestamps based on {sample_rate} Hz sample rate", verbose)
    
    # Save to HDF5
    try:
        # Remove the old file if it exists but is corrupted
        if hdf5_path.exists():
            try:
                hdf5_path.unlink()
                log_warning(f"Removed corrupted HDF5 file: {hdf5_path}", verbose)
            except Exception as e:
                log_error(f"Could not remove existing corrupted file: {str(e)}", verbose)
                # If we can't remove it, try a different name
                hdf5_path = backup_path.parent / f"{folder_name}-ArduinoDAQ_recovered.h5"
                log_warning(f"Will use alternative file name: {hdf5_path}", verbose)
        
        log_info(f"Creating HDF5 file: {hdf5_path}", verbose)
        with h5py.File(hdf5_path, 'w') as h5f:
            h5f.attrs['mouse_ID'] = mouse_ID
            h5f.attrs['date_time'] = date_time
            h5f.attrs['time'] = str(datetime.now())
            h5f.attrs['No_of_messages'] = num_messages
            h5f.attrs['reliability'] = 100.0
            h5f.attrs['skipped_long_messages'] = skipped_long_messages
            h5f.attrs['skipped_rows'] = skipped_rows if 'skipped_rows' in locals() else 0
            
            # Calculate time_taken from actual timestamps if available
            if has_timestamps and len(timestamps) > 1:
                time_taken = timestamps[-1] - timestamps[0]
            else:
                time_taken = num_messages / sample_rate
                
            h5f.attrs['time_taken'] = time_taken
            h5f.attrs['messages_per_second'] = sample_rate
            h5f.attrs['timestamp_source'] = "file" if has_timestamps else "generated"
            
            h5f.create_dataset('message_ids', data=message_ids, compression='gzip')
            h5f.create_dataset('timestamps', data=timestamps, compression='gzip')
            
            channel_group = h5f.create_group('channel_data')
            for idx, channel in enumerate(channel_indices):
                channel_group.create_dataset(channel, data=channel_data_array[:, idx], compression='gzip')
        
        # Verify the new file is valid
        if check_hdf5_integrity(hdf5_path, verbose):
            log_success(f"Successfully created and verified HDF5 file: {hdf5_path}", verbose)
            if skipped_long_messages > 0:
                log_warning(f"Note: File processed with {skipped_long_messages} skipped oversized messages", verbose)
            mark_as_processed(backup_path, "hdf5_conversion", verbose)
            return True
        else:
            log_error(f"Created HDF5 file but verification failed: {hdf5_path}", verbose)
            return False
            
    except Exception as e:
        log_error(f"Error saving HDF5 file: {str(e)}", verbose)
        return False

def process_session_folder(session_path, verbose=False):
    """
    Process a single session folder for both Arduino DAQ and camera frame ID recovery.
    
    Args:
        session_path (str or Path): Path to the session folder
        verbose (bool): Whether to print detailed information
    
    Returns:
        tuple: (num_processed_arduino, num_processed_tracker) - Count of successfully processed files
    """
    session_path = Path(session_path)
    log_info(f"\nProcessing session folder: {session_path}", verbose)
    
    num_processed_arduino = 0
    num_processed_tracker = 0
    
    # Handle the case where this is not a mouse folder but a logs folder
    if session_path.name.lower() == "logs":
        log_warning(f"Skipping logs directory: {session_path}", verbose)
        return num_processed_arduino, num_processed_tracker
        
    # Process Arduino DAQ backups
    arduino_backups = list(session_path.glob("*-backup.csv"))
    log_info(f"Found {len(arduino_backups)} Arduino backup files", verbose)
    for backup in arduino_backups:
        try:
            log_info(f"\nChecking Arduino backup: {backup}", verbose)
            if recover_from_backup(backup, verbose):
                log_success(f"Successfully processed Arduino backup: {backup}", verbose)
                num_processed_arduino += 1
            else:
                log_debug(f"No action needed for Arduino backup: {backup}", verbose)
        except Exception as e:
            log_error(f"Error processing Arduino backup {backup}: {str(e)}", verbose)
    
    # Process camera frame ID backups
    tracker_jsons = list(session_path.glob("*_Tracker_data.json"))
    log_info(f"Found {len(tracker_jsons)} tracker JSON files", verbose)

    # Check if backup exists but no JSON file exists
    backup_path = session_path / "frame_ids_backup.txt"
    if len(tracker_jsons) == 0 and backup_path.exists():
        log_info(f"No JSON files found but backup exists, creating one", verbose)
        # Create default JSON filename based on session path
        json_file = session_path / f"{session_path.name}_Tracker_data.json"
        # Create an empty JSON file 
        with open(json_file, 'w') as f:
            json.dump({}, f)
        tracker_jsons = [json_file]  # Add to the list to process

    for json_file in tracker_jsons:
        try:
            log_info(f"\nChecking camera data: {json_file}", verbose)
            if recover_frame_ids(json_file, verbose):
                log_success(f"Successfully recovered frame IDs: {json_file}", verbose)
                num_processed_tracker += 1
            else:
                log_debug(f"No action needed for camera data: {json_file}", verbose)
        except Exception as e:
            log_error(f"Error processing camera data {json_file}: {str(e)}", verbose)
    
    return num_processed_arduino, num_processed_tracker

def recover_crashed_sessions(directory, target_sessions=None, force=False, verbose=False):
    """
    Process all session folders in a cohort directory, including mouse subfolders.
    Structure: cohort_folder/session_folder/mouse_folder/
    
    Args:
        directory (str or Path): Path to cohort directory
        target_sessions (list): Optional list of session IDs to process (e.g., ['241017_151132'])
                               If provided, only these sessions will be processed
        force (bool): If True, reprocess files even if they have marker files
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: Summary of processed files
    """
    results = {
        'arduino_processed': 0,
        'tracker_processed': 0,
        'errors': 0,
        'sessions_found': 0,
        'sessions_processed': 0
    }
    
    try:
        directory = Path(directory)
        if not directory.exists():
            log_error(f"Directory not found: {directory}", verbose)
            return results
        
        # If force is True, remove all marker files first, optionally only for target sessions
        if force:
            log_warning("Force mode enabled - removing marker files", verbose)
            
            # Define a function to check if a marker file belongs to a target session
            def is_target_marker(marker_path):
                if target_sessions is None:
                    return True  # Process all if no targets specified
                
                marker_str = str(marker_path)
                return any(session_id in marker_str for session_id in target_sessions)
            
            # Find and remove marker files
            marker_files = list(directory.glob("**/*_hdf5_conversion_processed")) + \
                          list(directory.glob("**/*_frame_id_recovery_processed"))
            
            # Filter to only target sessions if specified
            marker_files = [m for m in marker_files if is_target_marker(m)]
            
            for marker in marker_files:
                try:
                    marker.unlink()
                    log_debug(f"Removed marker file: {marker}", verbose)
                except Exception as e:
                    log_error(f"Could not remove marker file {marker}: {str(e)}", verbose)
                    results['errors'] += 1
        
        # Define a function to check if a folder is a target session
        def is_target_session(folder_path):
            if target_sessions is None:
                return True  # Process all if no targets specified
            
            folder_str = str(folder_path)
            return any(session_id in folder_str for session_id in target_sessions)
        
        # Check if this is a session folder directly (files present)
        has_backups = len(list(directory.glob("*-backup.csv"))) > 0
        has_jsons = len(list(directory.glob("*_Tracker_data.json"))) > 0
        
        if (has_backups or has_jsons) and is_target_session(directory):
            log_info(f"\nDirectory appears to be a target session folder: {directory}", verbose)
            results['sessions_found'] += 1
            arduino_count, tracker_count = process_session_folder(directory, verbose)
            results['arduino_processed'] += arduino_count
            results['tracker_processed'] += tracker_count
            if arduino_count > 0 or tracker_count > 0:
                results['sessions_processed'] += 1
            return results
        
        # Otherwise, look for session folders (first level subdirectories)
        session_folders = [f for f in directory.iterdir() if f.is_dir()]
        log_info(f"Found {len(session_folders)} subfolders to process", verbose)
        
        for session_folder in session_folders:
            # Skip if not a target session
            if not is_target_session(session_folder):
                log_debug(f"Skipping non-target session: {session_folder}", verbose)
                continue
                
            # Check if this is a session folder with files
            has_session_backups = len(list(session_folder.glob("*-backup.csv"))) > 0
            has_session_jsons = len(list(session_folder.glob("*_Tracker_data.json"))) > 0
            
            if has_session_backups or has_session_jsons:
                log_info(f"\nProcessing target session folder: {session_folder.name}", verbose)
                results['sessions_found'] += 1
                arduino_count, tracker_count = process_session_folder(session_folder, verbose)
                results['arduino_processed'] += arduino_count
                results['tracker_processed'] += tracker_count
                if arduino_count > 0 or tracker_count > 0:
                    results['sessions_processed'] += 1
            else:
                log_info(f"\nProcessing subfolder: {session_folder.name}", verbose)
                
                # Check for special case of logs folder
                if session_folder.name.lower() == "logs":
                    log_warning(f"Skipping logs directory: {session_folder}", verbose)
                    continue
                    
                # Find all mouse folders within this session folder
                mouse_folders = [f for f in session_folder.iterdir() if f.is_dir()]
                
                # Filter mouse folders to only target sessions if specified
                target_mouse_folders = [f for f in mouse_folders if is_target_session(f)]
                
                if target_mouse_folders:
                    log_info(f"Found {len(target_mouse_folders)} target mouse folders in {session_folder.name}", verbose)
                else:
                    log_debug(f"No target mouse folders found in {session_folder.name}", verbose)
                    continue
                
                for mouse_folder in target_mouse_folders:
                    if mouse_folder.name.lower() == "logs":
                        log_warning(f"Skipping logs directory: {mouse_folder}", verbose)
                        continue
                        
                    log_info(f"\nProcessing target mouse folder: {mouse_folder.name}", verbose)
                    results['sessions_found'] += 1
                    arduino_count, tracker_count = process_session_folder(mouse_folder, verbose)
                    results['arduino_processed'] += arduino_count
                    results['tracker_processed'] += tracker_count
                    if arduino_count > 0 or tracker_count > 0:
                        results['sessions_processed'] += 1
        
        if verbose:
            log_success("\nRecovery process summary:")
            log_success(f"  Target sessions found: {results['sessions_found']}")
            log_success(f"  Target sessions processed: {results['sessions_processed']}")
            log_success(f"  Arduino DAQ files processed: {results['arduino_processed']}")
            log_success(f"  Tracker JSON files processed: {results['tracker_processed']}")
            log_warning(f"  Errors encountered: {results['errors']}")
            
        return results
            
    except Exception as e:
        log_error(f"Error processing cohort folder: {str(e)}", verbose)
        results['errors'] += 1
        return results

# Example usage if imported:
# from recovery_script import recover_crashed_sessions
# results = recover_crashed_sessions("E:/Data/Cohort5", force=False, verbose=True)
# print(f"Processed {results['arduino_processed']} Arduino files and {results['tracker_processed']} tracker files")

def main():
    # Manual usage of script:

    cohort_directory = r"/cephfs2/srogers/Behaviour code/2409_September_cohort/DATA_ArduinoDAQ"
    force_reprocess = True  # Set to True to force reprocessing of all files
    verbose_mode = True  # Set to True for detailed output

    # target_sessions = ["241017_151131"]  # Example target sessions
    target_sessions = None
    results = recover_crashed_sessions(cohort_directory, force=force_reprocess, verbose=verbose_mode, target_sessions=target_sessions)
    print(f"Processed {results['arduino_processed']} Arduino files and {results['tracker_processed']} tracker files")

if __name__ == "__main__":
    main()