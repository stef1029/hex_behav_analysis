import os
import csv
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def recover_frame_ids(json_file_path):
    """
    Recovers frame IDs from backup file if the JSON file has empty frame_IDs.
    """
    try:
        json_path = Path(json_file_path)
        if not json_path.exists():
            print(f"JSON file not found: {json_file_path}")
            return False
            
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Check if frame_IDs is empty
        if data.get('frame_IDs') and len(data['frame_IDs']) > 0:
            print(f"Frame IDs already exist in {json_path.name}, skipping...")
            return False
            
        # Construct backup file path
        base_name = json_path.stem.replace('_Tracker_data', '')
        backup_path = json_path.parent / f"{base_name}_frame_ids_backup.txt"
        
        if not backup_path.exists():
            print(f"Backup file not found: {backup_path}")
            return False
            
        print(f"\nProcessing {json_path.name}")
        print(f"Reading backup from: {backup_path.name}")
        
        # Read frame IDs from backup file
        frame_ids = []
        with open(backup_path, 'r') as f:
            for line in f:
                try:
                    frame_id = int(line.strip())
                    frame_ids.append(frame_id)
                except ValueError as e:
                    print(f"Warning: Invalid frame ID in backup file: {line.strip()}")
                    continue
        
        if not frame_ids:
            print("No valid frame IDs found in backup file")
            return False
            
        print(f"Found {len(frame_ids)} frame IDs")
        
        # Update the JSON data
        data['frame_IDs'] = frame_ids
        
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully updated {json_path.name} with {len(frame_ids)} frame IDs")
        return True
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")
        return False

def recover_from_backup(backup_file_path):
    """
    Recovers data from a backup CSV file and converts it to proper HDF5 format.
    Only processes the backup if no corresponding HDF5 file exists.
    """
    backup_path = Path(backup_file_path)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file_path}")
        
    # Check if HDF5 file already exists
    folder_name = backup_path.stem.replace('-backup', '')
    hdf5_path = backup_path.parent / f"{folder_name}-ArduinoDAQ.h5"
    if hdf5_path.exists():
        print(f"Skipping {backup_path} - HDF5 file already exists")
        return False
    
    # Extract folder name and mouse ID from the backup file name
    parts = folder_name.split('_')
    date_time = '_'.join(parts[:2])
    mouse_ID = '_'.join(parts[2:])
    
    # Read the backup CSV file
    messages_from_arduino = []
    try:
        print(f"\nProcessing backup file: {backup_path}")
        print(f"Date time: {date_time}")
        print(f"Mouse ID: {mouse_ID}")
        
        with open(backup_path, 'r') as csvfile:
            # Read first few lines to debug
            first_lines = []
            for i, line in enumerate(csvfile):
                if i < 5:
                    first_lines.append(line.strip())
            print(f"First few lines of data:")
            for line in first_lines:
                print(f"  {line}")
            
            # Reset file pointer to beginning
            csvfile.seek(0)
            
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) != 2:
                    print(f"Unexpected row format: {row}")
                    continue
                try:
                    message_id = int(row[0])
                    message_data = int(row[1])
                    messages_from_arduino.append([message_id, message_data])
                except ValueError as e:
                    print(f"Error parsing row {row}: {e}")
                    continue
                    
        if not messages_from_arduino:
            raise ValueError("No valid data could be parsed from the CSV file")
    except Exception as e:
        print(f"Error reading backup file: {str(e)}")
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
    
    # Process data
    message_ids = np.array([message[0] for message in messages_from_arduino], dtype=np.uint32)
    message_data = np.array([message[1] for message in messages_from_arduino], dtype=np.uint64)
    channel_data_array = np.zeros((num_messages, num_channels), dtype=np.uint8)
    
    for i, message in enumerate(message_data):
        binary_message = np.array(list(np.binary_repr(message, width=num_channels)), dtype=np.uint8)
        binary_message = binary_message[::-1]
        channel_data_array[i] = binary_message
    
    sample_rate = 1000
    timestamps = np.arange(num_messages) / sample_rate
    
    # Save to HDF5
    try:
        with h5py.File(hdf5_path, 'w') as h5f:
            h5f.attrs['mouse_ID'] = mouse_ID
            h5f.attrs['date_time'] = date_time
            h5f.attrs['time'] = str(datetime.now())
            h5f.attrs['No_of_messages'] = num_messages
            h5f.attrs['reliability'] = 100.0
            h5f.attrs['time_taken'] = num_messages / sample_rate
            h5f.attrs['messages_per_second'] = sample_rate
            
            h5f.create_dataset('message_ids', data=message_ids, compression='gzip')
            h5f.create_dataset('timestamps', data=timestamps, compression='gzip')
            
            channel_group = h5f.create_group('channel_data')
            for idx, channel in enumerate(channel_indices):
                channel_group.create_dataset(channel, data=channel_data_array[:, idx], compression='gzip')
                
        return True
    except Exception as e:
        print(f"Error saving HDF5 file: {str(e)}")
        return False

def process_session_folder(session_path):
    """
    Process a single session folder for both Arduino DAQ and camera frame ID recovery.
    """
    session_path = Path(session_path)
    print(f"\nProcessing session folder: {session_path}")
    
    # Process Arduino DAQ backups
    arduino_backups = list(session_path.glob("*-backup.csv"))
    for backup in arduino_backups:
        try:
            print(f"\nChecking Arduino backup: {backup}")
            if recover_from_backup(backup):
                print(f"Successfully processed Arduino backup: {backup}")
        except Exception as e:
            print(f"Error processing Arduino backup {backup}: {str(e)}")
    
    # Process camera frame ID backups
    tracker_jsons = list(session_path.glob("*_Tracker_data.json"))
    for json_file in tracker_jsons:
        try:
            print(f"\nChecking camera data: {json_file}")
            if recover_frame_ids(json_file):
                print(f"Successfully recovered frame IDs: {json_file}")
        except Exception as e:
            print(f"Error processing camera data {json_file}: {str(e)}")

def recover_crashed_sessions(directory):
    """
    Process all session folders in a cohort directory, including mouse subfolders.
    Structure: cohort_folder/session_folder/mouse_folder/
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return
        
        # Find all session folders (first level subdirectories)
        session_folders = [f for f in directory.iterdir() if f.is_dir()]
        print(f"Found {len(session_folders)} session folders to process")
        
        for session_folder in session_folders:
            print(f"\nProcessing session folder: {session_folder.name}")
            
            # Find all mouse folders within this session folder
            mouse_folders = [f for f in session_folder.iterdir() if f.is_dir()]
            print(f"Found {len(mouse_folders)} mouse folders in {session_folder.name}")
            
            for mouse_folder in mouse_folders:
                print(f"\nProcessing mouse folder: {mouse_folder.name}")
                process_session_folder(mouse_folder)
            
    except Exception as e:
        print(f"Error processing cohort folder: {str(e)}")


if __name__ == "__main__":

    path = r"E:\Dan_test"
    
    recover_crashed_sessions(path)
