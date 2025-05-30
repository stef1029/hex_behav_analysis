#!/usr/bin/env python3
# event_parser.py - Parses binary event data and creates timestamp files
import sys
import struct
import base64
import json
import os
import h5py
import numpy as np
import traceback
from typing import List, Tuple, Dict, Optional, Union

# Import the analysis module
from hex_behav_analysis.ephys.ephys_sync_timestamp_analysis import analyze_timestamps, save_analysis_plots
from hex_behav_analysis.ephys.debug_inp_files import inspect_file

def parse_binary_events(file_path: str, target_pin: int, verbose: bool = False) -> List[Tuple[float, float]]:
    """
    Parse a binary file with event data and extract events for a specific pin.
    
    Args:
        file_path (str): Path to the binary data file
        target_pin (int): The pin number to extract events for (0-15)
        verbose (bool): Whether to print detailed information during processing
    
    Returns:
        List[Tuple[float, float]]: List of tuples containing (high_timestamp, duration) for the target pin
        high_timestamp is the time when signal went high (in seconds), duration is how long it remained high (in seconds)
    """
    # Read the file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Check if the file is base64 encoded
    try:
        # Try to decode a small chunk to test if it's base64
        test_chunk = content[:100]
        decoded_test = base64.b64decode(test_chunk)
        # If we got here, it's probably base64 encoded
        content = base64.b64decode(content)
    except:
        # Not base64 encoded, use as is
        pass
    
    # Convert to string for header parsing
    content_str = content.decode('latin-1', errors='replace')
    
    # Find the start of binary data
    data_start_index = content_str.lower().find("data_start".lower())
    if data_start_index == -1:
        # Debug the file content
        print(f"File size: {len(content)} bytes")
        print(f"First 200 characters: {content_str[:200]}")
        # List potential markers that might be present
        possible_markers = ["data", "start", "begin", "header"]
        for marker in possible_markers:
            pos = content_str.find(marker)
            if pos != -1:
                print(f"Found '{marker}' at position {pos}")
        raise ValueError("Could not find 'data_start' marker in the file")
    
    # Extract and parse header
    header_str = content_str[:data_start_index]
    header_lines = [line.strip() for line in header_str.splitlines() if line.strip()]
    
    if verbose:
        print(f"Found data_start marker at position {data_start_index}")
        print(f"Header length: {len(header_str)} bytes")
    
    # Parse metadata from header
    metadata = {}
    for line in header_lines:
        parts = line.split(None, 1)  # Split on first whitespace
        if len(parts) >= 2:
            metadata[parts[0]] = parts[1].strip()
        elif len(parts) == 1:
            metadata[parts[0]] = ""
    
    # Extract relevant metadata
    bytes_per_sample = int(metadata.get('bytes_per_sample', 7))
    bytes_per_timestamp = int(metadata.get('bytes_per_timestamp', 4))
    bytes_per_type = int(metadata.get('bytes_per_type', 1))
    bytes_per_value = int(metadata.get('bytes_per_value', 2))
    timebase = int(metadata.get('timebase', '1000 hz').split()[0])
    
    # Get binary data after the header + "data_start" marker
    data_start_pos = data_start_index + len("data_start")
    
    # Check if there's a data_end marker
    data_end_index = content_str.find("data_end")
    if data_end_index != -1:
        # Extract only the data between start and end markers
        binary_data = content[data_start_pos:data_end_index]
        if verbose:
            print(f"Found data_end marker at position {data_end_index}")
    else:
        # No end marker, use all remaining data
        binary_data = content[data_start_pos:]
    
    if verbose:
        print(f"File: {file_path}")
        print(f"Metadata: {metadata}")
        print(f"Targeting pin: {target_pin}")
        print(f"Timebase: {timebase} Hz")
        print(f"Bytes per sample: {bytes_per_sample}")
        print(f"Total binary data size: {len(binary_data)} bytes")
        print(f"Expected number of samples: ~{len(binary_data) // bytes_per_sample}")
    
    # Parse events
    raw_events = []
    previous_state = None
    
    # For each sample in the binary data
    for i in range(0, len(binary_data), bytes_per_sample):
        if i + bytes_per_sample <= len(binary_data):
            try:
                # Extract timestamp (4 bytes, big-endian)
                timestamp_bytes = binary_data[i:i+bytes_per_timestamp]
                if len(timestamp_bytes) < bytes_per_timestamp:
                    if verbose:
                        print(f"Warning: Incomplete timestamp data at position {i}, skipping")
                    continue
                timestamp = struct.unpack('>I', timestamp_bytes)[0]
                
                # Extract type (1 byte)
                if i+bytes_per_timestamp >= len(binary_data):
                    continue
                type_byte = binary_data[i+bytes_per_timestamp]
                
                # Extract value (2 bytes, big-endian)
                value_offset = i+bytes_per_timestamp+bytes_per_type
                if value_offset + bytes_per_value > len(binary_data):
                    if verbose:
                        print(f"Warning: Incomplete value data at position {i}, skipping")
                    continue
                    
                value_bytes = binary_data[value_offset:value_offset+bytes_per_value]
                value = struct.unpack('>H', value_bytes)[0]
                
                # Sanity check - ignore extremely large values that might be from parsing metadata
                if timestamp > 1000000000:  # More than ~11.5 days at 1000Hz
                    if verbose:
                        print(f"Warning: Suspiciously large timestamp at position {i}, skipping")
                    continue
            except Exception as e:
                if verbose:
                    print(f"Error parsing data at position {i}: {e}")
                continue
            
            # Convert timestamp to seconds
            time_in_seconds = timestamp / timebase
            
            # Check if the target pin is active
            pin_state = bool(value & (1 << target_pin))
            
            # Record pin state changes
            if previous_state is None or pin_state != previous_state:
                raw_events.append((time_in_seconds, pin_state))
                previous_state = pin_state
    
    # Convert raw events to high pulse events with durations
    pin_events = []
    for i in range(len(raw_events) - 1):
        current_time, current_state = raw_events[i]
        next_time, next_state = raw_events[i + 1]
        
        # Only process events where signal went from low to high
        if current_state == True and (i == 0 or raw_events[i-1][1] == False):
            # Calculate duration until signal goes low
            duration = next_time - current_time
            pin_events.append((current_time, duration))
    
    # Handle the case where the signal is still high at the end of the recording
    if raw_events and raw_events[-1][1] == True:
        if verbose:
            print("Warning: Signal was still high at the end of the recording. Estimating duration.")
        # Use a minimal duration for the last high pulse if no matching low transition
        if len(raw_events) >= 2:
            # Estimate a duration based on previous pulse durations
            avg_duration = 0
            count = 0
            for i in range(len(pin_events)):
                avg_duration += pin_events[i][1]
                count += 1
            
            if count > 0:
                avg_duration = avg_duration / count
                pin_events.append((raw_events[-1][0], avg_duration))
            else:
                # Fallback to a small duration if no previous pulses to average
                pin_events.append((raw_events[-1][0], 0.001))  # 1ms default duration
    
    return pin_events

def get_metadata(file_path: str) -> Dict[str, str]:
    with open(file_path, 'rb') as f:
        content = f.read()

    # inspect_file(file_path)  # Inspect the file for encoding issues
    
    # Check if the file is base64 encoded
    try:
        test_chunk = content[:100]
        decoded_test = base64.b64decode(test_chunk)
        content = base64.b64decode(content)
    except:
        pass
    
    # Always use latin-1 encoding, which can handle any byte value
    content_str = content.decode('utf-8', errors='replace')
    
    data_start_index = content_str.find("data_start")
    
    if data_start_index == -1:
        raise ValueError("Could not find 'data_start' marker in the file")
    
    header_str = content_str[:data_start_index]
    header_lines = [line.strip() for line in header_str.splitlines() if line.strip()]
    
    metadata = {}
    for line in header_lines:
        parts = line.split(None, 1)
        if len(parts) >= 2:
            metadata[parts[0]] = parts[1].strip()
        elif len(parts) == 1:
            metadata[parts[0]] = ""
    
    return metadata

def parse_all_pins(file_path: str, verbose: bool = False) -> Dict[int, List[Tuple[float, float]]]:
    """
    Parse a binary file with event data and extract events for all pins.
    
    Args:
        file_path (str): Path to the binary data file
        verbose (bool): Whether to print detailed information
    
    Returns:
        Dict[int, List[Tuple[float, float]]]: Dictionary mapping pin numbers to lists
            of (high_timestamp, duration) tuples
    """
    all_pin_events = {}
    # Assuming 16 pins (0-15) in a 2-byte value
    for pin in range(16):
        events = parse_binary_events(file_path, pin, verbose=False)
        if events:  # Only include pins that have events
            all_pin_events[pin] = events
            if verbose:
                print(f"Pin {pin}: Found {len(events)} events")
    return all_pin_events

def save_to_json(data: Dict[int, List[Tuple[float, float]]], metadata: Dict[str, str], 
                 analysis: Dict[int, Dict[str, float]], output_path: str) -> None:
    """
    Save parsed event data, metadata, and analysis to a JSON file.
    
    Args:
        data (Dict[int, List[Tuple[float, float]]]): Dictionary mapping pin numbers to event lists
        metadata (Dict[str, str]): Metadata dictionary
        analysis (Dict[int, Dict[str, float]]): Analysis results for each pin
        output_path (str): Path to save the JSON file
    """
    # Merge analysis results into metadata
    combined_metadata = metadata.copy()
    for pin, analysis_data in analysis.items():
        for key, value in analysis_data.items():
            combined_metadata[f"pin_{pin}_{key}"] = value
    
    # Convert the data structure to be JSON serializable
    json_data = {
        "metadata": combined_metadata,
        "pins": {},
        "analysis": analysis
    }
    
    for pin, events in data.items():
        # Convert each event to a dictionary with high_timestamp and duration
        json_data["pins"][str(pin)] = [
            {"high_timestamp": high_time, "duration": duration}
            for high_time, duration in events
        ]
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

def save_to_hdf5(data: Dict[int, List[Tuple[float, float]]], metadata: Dict[str, str], 
                 analysis: Dict[int, Dict[str, float]], output_path: str) -> None:
    """
    Save parsed event data, metadata, and analysis to an HDF5 file.
    
    Args:
        data (Dict[int, List[Tuple[float, float]]]): Dictionary mapping pin numbers to event lists
        metadata (Dict[str, str]): Metadata dictionary
        analysis (Dict[int, Dict[str, float]]): Analysis results for each pin
        output_path (str): Path to save the HDF5 file
    """
    with h5py.File(output_path, 'w') as f:
        # Create a metadata group
        meta_group = f.create_group('metadata')
        for key, value in metadata.items():
            meta_group.attrs[key] = str(value)
        
        # Create an analysis group
        analysis_group = f.create_group('analysis')
        for pin, analysis_data in analysis.items():
            pin_analysis = analysis_group.create_group(f'pin_{pin}')
            for key, value in analysis_data.items():
                pin_analysis.attrs[key] = value
        
        # Store each pin's data
        pins_group = f.create_group('pins')
        for pin, events in data.items():
            if not events:
                continue
                
            # Convert events to separate arrays for high timestamps and durations
            high_timestamps = np.array([event[0] for event in events], dtype=np.float64)
            durations = np.array([event[1] for event in events], dtype=np.float64)
            
            # Create a dataset for this pin
            pin_group = pins_group.create_group(f'pin_{pin}')
            pin_group.create_dataset('high_timestamps', data=high_timestamps)
            pin_group.create_dataset('durations', data=durations)

def get_session_id_from_output_folder(output_folder: str) -> str:
    """
    Extract the session ID from the output folder path.
    
    Args:
        output_folder (str): The output folder path
    
    Returns:
        str: The extracted session ID
    """
    # Get the folder name (last part of the path)
    folder_name = os.path.basename(os.path.normpath(output_folder))
    return folder_name

def process_file(file_path: str, output_folder: str, target_pin: Optional[int] = None, 
                generate_plots: bool = False, verbose: bool = False) -> None:
    """
    Process a binary event file, analyze timestamps, and save the results.
    
    Args:
        file_path (str): Path to the binary data file
        output_folder (str): Folder to save the output files
        target_pin (Optional[int]): The specific pin to process, or None to process all pins
        generate_plots (bool): Whether to generate analysis plots
        verbose (bool): Whether to print detailed information during processing
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract session ID from the output folder name
    session_id = get_session_id_from_output_folder(output_folder)
    
    # Define output filenames with the required structure
    output_filename = f"{session_id}_ephys_sync_timestamps"
    
    # Extract metadata
    metadata = get_metadata(file_path)
    
    if verbose:
        print(f"Processing file: {file_path}")
        print(f"Output folder: {output_folder}")
        print(f"Session ID: {session_id}")
        print(f"Output filename base: {output_filename}")
        print(f"Metadata: {metadata}")
    
    # Parse event data
    if target_pin is not None:
        # Process single pin
        events = parse_binary_events(file_path, target_pin, verbose=verbose)
        all_pin_events = {target_pin: events}
    else:
        # Process all pins
        all_pin_events = parse_all_pins(file_path, verbose=verbose)
    
    if verbose:
        print(f"Processed {len(all_pin_events)} pins with events")
    
    # Analyze timestamps for each pin
    all_pin_analysis = {}
    for pin, events in all_pin_events.items():
        if len(events) > 1:  # Need at least 2 events to analyze timing
            pin_analysis = analyze_timestamps(events)
            all_pin_analysis[pin] = pin_analysis
            
            if verbose:
                print(f"\nAnalysis for Pin {pin}:")
                for key, value in pin_analysis.items():
                    print(f"  {key}: {value}")
    
    # Add summary to file metadata
    for pin, analysis in all_pin_analysis.items():
        metadata[f"pin_{pin}_pulse_count"] = str(analysis.get("pulse_count", 0))
        metadata[f"pin_{pin}_mean_interval_ms"] = f"{analysis.get('mean_interval_ms', 0):.4f}"
        metadata[f"pin_{pin}_missed_pulses"] = str(analysis.get("missed_pulse_count", 0))
        metadata[f"pin_{pin}_missed_pulse_percent"] = f"{analysis.get('missed_pulse_percent', 0):.4f}%"
        metadata[f"pin_{pin}_jitter_max_ms"] = f"{analysis.get('jitter_max_ms', 0):.4f}"
        metadata[f"pin_{pin}_drift_rate_ms_per_s"] = f"{analysis.get('drift_rate_ms_per_s', 0):.6f}"
    
    # Define output paths with the naming scheme
    json_path = os.path.join(output_folder, f"{output_filename}.json")
    hdf5_path = os.path.join(output_folder, f"{output_filename}.h5")
    
    # Save the data
    save_to_json(all_pin_events, metadata, all_pin_analysis, json_path)
    save_to_hdf5(all_pin_events, metadata, all_pin_analysis, hdf5_path)
    
    # Generate analysis plots if requested
    if generate_plots:
        save_analysis_plots(all_pin_events, all_pin_analysis, output_folder, session_id, target_pin)
    
    if verbose:
        print(f"Saved JSON data to: {json_path}")
        print(f"Saved HDF5 data to: {hdf5_path}")
        
        # Print summary information
        total_events = sum(len(events) for events in all_pin_events.values())
        print(f"Total events saved: {total_events}")
        
        if generate_plots:
            print(f"Analysis plots saved to: {os.path.join(output_folder, 'analysis_plots')}")

def main():
    """
    Command line interface for the script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse binary event data, analyze timing, and save results.')
    parser.add_argument('file_path', help='Path to the binary event data file')
    parser.add_argument('--output-folder', '-o', default='./output', help='Folder to save output files (default: ./output)')
    parser.add_argument('--pin', '-p', type=int, help='Specific pin to process (0-15, default: process all pins)')
    parser.add_argument('--plots', '-g', action='store_true', help='Generate analysis plots')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        process_file(args.file_path, args.output_folder, args.pin, args.plots, args.verbose)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()