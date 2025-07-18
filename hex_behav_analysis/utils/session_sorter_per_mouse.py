from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from colorama import Fore, Style, init
from typing import Dict, List, Optional, Tuple

init()


def calculate_total_time_seconds(experiment_data: dict) -> float:
    """
    Calculate the total time in seconds between the start and end times in the experiment data.
    
    Args:
        experiment_data: Dictionary containing experiment metadata with 'Date and time' 
                        and 'End time' fields.
    
    Returns:
        Total time in seconds.
    
    Notes:
        - 'Date and time' format is expected to be 'YYMMDD_HHMMSS'
        - 'End time' format is expected to be 'YYMMDDHHMMSS'
    """
    try:
        start_time_str = experiment_data["Date and time"]
        end_time_str = experiment_data["End time"]
        
        # Parse the start time (Format: YYMMDD_HHMMSS)
        start_year = int("20" + start_time_str[0:2])
        start_month = int(start_time_str[2:4])
        start_day = int(start_time_str[4:6])
        start_hour = int(start_time_str[7:9])
        start_minute = int(start_time_str[9:11])
        start_second = int(start_time_str[11:13])
        
        # Parse the end time (Format: YYMMDDHHMMSS)
        end_year = int("20" + end_time_str[0:2])
        end_month = int(end_time_str[2:4])
        end_day = int(end_time_str[4:6])
        end_hour = int(end_time_str[6:8])
        end_minute = int(end_time_str[8:10])
        end_second = int(end_time_str[10:12])
        
        # Calculate total seconds
        start_datetime = datetime(
            start_year, start_month, start_day, 
            start_hour, start_minute, start_second
        )
        
        end_datetime = datetime(
            end_year, end_month, end_day,
            end_hour, end_minute, end_second
        )
        
        time_difference = (end_datetime - start_datetime).total_seconds()
        
        return time_difference
    except (KeyError, ValueError) as e:
        print(f"Error calculating time: {e}")
        return 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "45m 30s")
    """
    if seconds <= 0:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_comprehensive_metadata(session_id: str, cohort: Cohort_folder) -> Dict:
    """
    Extract comprehensive metadata for a given session including duration, trial count, 
    cue duration, phase, audio trials, and ports in use.
    
    Args:
        session_id: Unique identifier for the session
        cohort: The cohort object containing session data
    
    Returns:
        Dictionary containing all session metadata
    """
    session_dict = cohort.get_session(session_id)
    if session_dict is None:
        return {
            "total_time": 0,
            "total_time_formatted": "N/A",
            "num_trials": "N/A",
            "cue_duration": "N/A",
            "wait_duration": "N/A",
            "phase": "N/A",
            "audio_trials": False,
            "ports_in_use": "N/A",
            "error": "Session not found"
        }
    
    behaviour_data_path = session_dict.get("raw_data", {}).get("behaviour_data", {})
    
    try:
        with open(behaviour_data_path, 'r') as f:
            metadata = json.load(f)
        
        # Calculate session duration
        total_time = calculate_total_time_seconds(metadata)
        
        # Extract ports in use - check both possible keys
        ports = metadata.get("Ports in use", None)
        if ports is None:
            ports = metadata.get("Port", None)
            
        if ports:
            # Format ports as a readable string
            if isinstance(ports, list):
                ports_sorted = sorted(ports)
                # Always use comma-separated format to avoid Excel date conversion
                # Or use a prefix to ensure Excel treats it as text
                ports_str = "P:" + ",".join(map(str, ports_sorted))
            else:
                ports_str = f"P:{ports}"
        else:
            ports_str = "N/A"
        
        # Extract all relevant information
        result = {
            "total_time": total_time,
            "total_time_formatted": format_duration(total_time),
            "num_trials": metadata.get("Total trials", "N/A"),
            "cue_duration": metadata.get("Cue duration", "N/A"),
            "wait_duration": metadata.get("Wait duration", "N/A"),
            "phase": metadata.get("Behaviour phase", "N/A"),
            "audio_trials": "Audio Trials" in metadata,
            "ports_in_use": ports_str,
            "error": None
        }
        
        return result
        
    except Exception as e:
        return {
            "total_time": 0,
            "total_time_formatted": "N/A",
            "num_trials": "N/A",
            "cue_duration": "N/A",
            "wait_duration": "N/A",
            "phase": "N/A",
            "audio_trials": False,
            "ports_in_use": "N/A",
            "error": str(e)
        }


def format_session_info(session_id: str, metadata: Dict) -> str:
    """
    Format session information into a readable string.
    
    Args:
        session_id: The session identifier
        metadata: Dictionary containing session metadata
        
    Returns:
        Formatted string with session information
    """
    parts = []
    
    # Add phase if available
    if metadata["phase"] != "N/A":
        parts.append(f"Phase: {metadata['phase']}")
    
    # Add trial count
    parts.append(f"Trials: {metadata['num_trials']}")
    
    # Add duration
    parts.append(f"Duration: {metadata['total_time_formatted']}")
    
    # Add ports in use
    if metadata["ports_in_use"] != "N/A":
        parts.append(f"Ports: {metadata['ports_in_use']}")
    
    # Add cue/wait durations if available
    if metadata["cue_duration"] != "N/A" and str(metadata["cue_duration"]).strip():
        parts.append(f"Cue: {metadata['cue_duration']}ms")
    
    if metadata["wait_duration"] != "N/A" and str(metadata["wait_duration"]).strip():
        parts.append(f"Wait: {metadata['wait_duration']}ms")
    
    # Add audio trials indicator
    if metadata["audio_trials"]:
        parts.append("[AUDIO]")
    
    # Combine all parts
    info_str = " | ".join(parts)
    
    return f"{session_id}: {info_str}"


def print_mouse_sessions(mouse_id: str, sessions: List[Tuple[str, Dict]], colour: str = Fore.WHITE, 
                        min_duration: Optional[float] = None):
    """
    Print all sessions for a specific mouse in a formatted way.
    
    Args:
        mouse_id: The mouse identifier
        sessions: List of tuples containing (session_id, metadata)
        colour: Colorama colour for the mouse header
        min_duration: Minimum duration in seconds (sessions below this are filtered out)
    """
    # Filter sessions by duration if specified
    filtered_sessions = sessions
    filtered_count = 0
    
    if min_duration is not None:
        filtered_sessions = [
            (sid, meta) for sid, meta in sessions 
            if meta["total_time"] >= min_duration or meta["total_time"] == 0  # Keep errors
        ]
        filtered_count = len(sessions) - len(filtered_sessions)
    
    print(f"\n{colour}{Style.BRIGHT}{'='*80}")
    print(f"MOUSE: {mouse_id}")
    print(f"Total sessions: {len(filtered_sessions)}")
    if filtered_count > 0:
        print(f"Filtered out: {filtered_count} sessions (duration < {min_duration}s)")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Sort sessions by date/time
    sorted_sessions = sorted(filtered_sessions, key=lambda x: x[0])
    
    # Group by date for better organisation
    current_date = None
    
    for session_id, metadata in sorted_sessions:
        session_date = session_id[:6]  # Extract YYMMDD
        
        # Print date header if new date
        if session_date != current_date:
            current_date = session_date
            # Convert to readable date format
            try:
                date_obj = datetime.strptime(f"20{session_date}", "%Y%m%d")
                date_str = date_obj.strftime("%A, %d %B %Y")
                print(f"\n  {Style.BRIGHT}Date: {date_str}{Style.RESET_ALL}")
            except ValueError:
                print(f"\n  {Style.BRIGHT}Date: {session_date}{Style.RESET_ALL}")
        
        # Format and print session info
        session_info = format_session_info(session_id, metadata)
        
        # Highlight certain conditions
        if metadata["audio_trials"]:
            print(f"    {Fore.YELLOW}{session_info}{Style.RESET_ALL}")
        elif metadata["error"]:
            print(f"    {Fore.RED}{session_info} [ERROR: {metadata['error']}]{Style.RESET_ALL}")
        else:
            print(f"    {session_info}")


def discover_all_mouse_sessions(cohort_folder_path: str) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Discover all sessions for each mouse in the cohort folder.
    
    Args:
        cohort_folder_path: Path to the cohort folder
        
    Returns:
        Dictionary mapping mouse IDs to lists of (session_id, metadata) tuples
    """
    print(f"Loading cohort from: {cohort_folder_path}")
    
    cohort = Cohort_folder(
        cohort_folder_path,
        multi=True,
        OEAB_legacy=False,
        use_existing_cohort_info=False,
        ignore_tests=True
    )
    
    phases = cohort.phases()
    
    # Organise sessions by mouse
    mouse_sessions = {}
    
    print("Discovering sessions...")
    total_sessions = 0
    
    for phase in phases:
        for session_id in phases[phase]:
            mouse_id = phases[phase][session_id]["mouse"]
            
            # Get comprehensive metadata for this session
            metadata = get_comprehensive_metadata(session_id, cohort)
            
            # Add to mouse's session list
            if mouse_id not in mouse_sessions:
                mouse_sessions[mouse_id] = []
            
            mouse_sessions[mouse_id].append((session_id, metadata))
            total_sessions += 1
    
    print(f"Found {total_sessions} total sessions across {len(mouse_sessions)} mice")
    
    return mouse_sessions


def export_to_csv(mouse_sessions: Dict[str, List[Tuple[str, Dict]]], output_path: str = "mouse_sessions.csv",
                  min_duration: Optional[float] = None):
    """
    Export session data to CSV for easier sorting and analysis.
    Sessions are sorted by date/time within each mouse, with spacing between mice.
    
    Args:
        mouse_sessions: Dictionary mapping mouse IDs to session data
        output_path: Path for the output CSV file
        min_duration: Minimum duration in seconds (sessions below this are filtered out)
    """
    rows = []
    total_filtered = 0
    
    # Sort mice by ID for consistent ordering
    sorted_mice = sorted(mouse_sessions.keys())
    
    for i, mouse_id in enumerate(sorted_mice):
        # Sort sessions for this mouse by session ID (which contains date/time)
        sorted_sessions = sorted(mouse_sessions[mouse_id], key=lambda x: x[0])
        
        # Filter by duration if specified
        filtered_count = 0
        if min_duration is not None:
            filtered_sessions = [
                (sid, meta) for sid, meta in sorted_sessions 
                if meta["total_time"] >= min_duration or meta["total_time"] == 0  # Keep errors
            ]
            filtered_count = len(sorted_sessions) - len(filtered_sessions)
            total_filtered += filtered_count
            sorted_sessions = filtered_sessions
        
        # Add mouse header comment row
        header_text = f"Total sessions: {len(sorted_sessions)}"
        if min_duration is not None and filtered_count > 0:
            header_text += f" (filtered: {filtered_count})"
            
        mouse_header = {
            "mouse_id": f"--- MOUSE: {mouse_id} ---",
            "session_id": header_text,
            "date": "",
            "time": "",
            "phase": "",
            "num_trials": "",
            "duration_seconds": "",
            "duration_formatted": "",
            "ports_in_use": "",
            "cue_duration_ms": "",
            "wait_duration_ms": "",
            "has_audio_trials": "",
            "error": ""
        }
        rows.append(mouse_header)
        
        # Add data rows for this mouse
        for session_id, metadata in sorted_sessions:
            row = {
                "mouse_id": mouse_id,
                "session_id": session_id,
                "date": session_id[:6],
                "time": session_id[7:13],
                "phase": metadata["phase"],
                "num_trials": metadata["num_trials"],
                "duration_seconds": metadata["total_time"],
                "duration_formatted": metadata["total_time_formatted"],
                "ports_in_use": metadata["ports_in_use"],
                "cue_duration_ms": metadata["cue_duration"],
                "wait_duration_ms": metadata["wait_duration"],
                "has_audio_trials": metadata["audio_trials"],
                "error": metadata["error"]
            }
            rows.append(row)
        
        # Add empty rows between mice (except after the last mouse)
        if i < len(sorted_mice) - 1:
            # Add two empty rows for spacing
            empty_row = {col: "" for col in row.keys()}
            rows.append(empty_row)
            rows.append(empty_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nExported session data to: {output_path}")
    if min_duration is not None and total_filtered > 0:
        print(f"Filtered out {total_filtered} sessions with duration < {min_duration}s")


def main():
    """
    Main function to discover and display all mouse sessions from a cohort folder.
    """
    # Configuration
    cohort_directory = "/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Training"  # Update this path
    export_csv = True  # Set to True to also export to CSV
    
    # Filtering parameters
    min_duration_seconds = 360  # Minimum session duration in seconds (5 minutes)
    # Set to None to include all sessions
    
    # Discover all sessions
    mouse_sessions = discover_all_mouse_sessions(cohort_directory)
    
    # Define colours for different mice (cycling through if more mice than colours)
    colours = [Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.RED, Fore.GREEN, Fore.YELLOW]
    
    # Sort mice by ID for consistent ordering
    sorted_mice = sorted(mouse_sessions.keys())
    
    # Print filter information if active
    if min_duration_seconds is not None:
        print(f"\n{Style.BRIGHT}Filter active: Excluding sessions < {min_duration_seconds}s ({min_duration_seconds/60:.1f} minutes){Style.RESET_ALL}")
    
    # Print sessions for each mouse
    for i, mouse_id in enumerate(sorted_mice):
        colour = colours[i % len(colours)]
        sessions = mouse_sessions[mouse_id]
        print_mouse_sessions(mouse_id, sessions, colour, min_duration_seconds)
    
    # Print summary statistics
    print(f"\n{Style.BRIGHT}{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Calculate statistics with filtering
    total_sessions = 0
    filtered_sessions = 0
    phase_counts = {}
    audio_count = 0
    
    for mouse_id, sessions in mouse_sessions.items():
        for session_id, metadata in sessions:
            # Apply duration filter for statistics
            if min_duration_seconds is None or metadata["total_time"] >= min_duration_seconds or metadata["total_time"] == 0:
                total_sessions += 1
                phase = metadata["phase"]
                if phase != "N/A":
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if metadata["audio_trials"]:
                    audio_count += 1
            else:
                filtered_sessions += 1
    
    print(f"Total mice: {len(mouse_sessions)}")
    print(f"Total sessions (after filtering): {total_sessions}")
    if filtered_sessions > 0:
        print(f"Sessions filtered out: {filtered_sessions}")
    
    print("\nSessions by phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} sessions")
    
    print(f"\nSessions with audio trials: {audio_count}")
    
    # Export to CSV if requested
    if export_csv:
        csv_path = Path(cohort_directory) / "mouse_sessions_summary.csv"
        export_to_csv(mouse_sessions, str(csv_path), min_duration_seconds)
    
    print("\nLegend:")
    print(f"  {Fore.YELLOW}Yellow sessions{Style.RESET_ALL} = Contains audio trials")
    print(f"  {Fore.RED}Red sessions{Style.RESET_ALL} = Error loading metadata")


if __name__ == "__main__":
    main()