from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
import json
from pathlib import Path
import pandas as pd
import re
from datetime import datetime


def get_behaviour_data(session_id, cohort):
    """
    Get the behaviour data file for a given session.
    """
    session_dict=cohort.get_session(session_id)
    file_path = Path(session_dict.get("raw_data").get("behaviour_data"))
    if file_path is not None and file_path.exists():
        with file_path.open("r") as f:
            bd = json.load(f)

    return bd

def analyze_session_groups(session_groups, cohort):
    """
    Analyze multiple session groups and return the proportion of attempted trials
    for each mouse in each condition.
    
    Args:
        session_groups (dict): Dictionary containing session groups with control and test sessions
        cohort (Cohort_folder): Cohort object to retrieve session data
        
    Returns:
        dict: Dictionary containing results for each session group with the following structure:
            - session_group_name (str):
                - 'control': Results from parse_mouse_trials for control sessions
                    - 'average': Average proportion across all mice in control condition
                    - 'mice': Dictionary with mouse IDs as keys and their average proportions as values
                - 'test': Results from parse_mouse_trials for test sessions
                    - 'average': Average proportion across all mice in test condition
                    - 'mice': Dictionary with mouse IDs as keys and their average proportions as values
    """
    # Dictionary to store results for each session group
    results = {}
    
    # Process each session group
    for session_group_name, session_data in session_groups.items():
        # Initialize results for this session group
        results[session_group_name] = {}
        
        # Process control sessions if they exist
        sessions_ctrl = session_data.get("sessions_ctrl", [])
        if sessions_ctrl:
            results[session_group_name]["control"] = parse_mouse_trials(sessions_ctrl, cohort)
        else:
            results[session_group_name]["control"] = None
        
        # Process test sessions if they exist
        sessions_test = session_data.get("sessions_test", [])
        if sessions_test:
            results[session_group_name]["test"] = parse_mouse_trials(sessions_test, cohort)
        else:
            results[session_group_name]["test"] = None
    
    return results

def parse_mouse_trials(session_ids, cohort):
    """
    Load and parse mouse trial data for multiple sessions and return the proportion of 
    attempted trials (successes + failures) to total trials for each mouse.
    
    Args:
        session_ids (list): List of session IDs to analyze
        cohort (Cohort_folder): Cohort object to retrieve session data
        
    Returns:
        dict: Dictionary containing:
            - 'average': Average proportion of attempts across all mice
            - 'mice': Dictionary with mouse IDs as keys and their average attempt proportions as values
    """
    # Dictionary to store results for each session and mouse
    session_results = {}
    mouse_sessions = {}
    
    # Extract mouse IDs from session IDs and organize sessions by mouse
    for session_id in session_ids:
        # Extract the mouse ID from the session ID (everything after the last underscore)
        parts = session_id.split('_')
        if len(parts) >= 3:
            mouse_id = parts[2]
            
            # Initialize the list for this mouse if it doesn't exist
            if mouse_id not in mouse_sessions:
                mouse_sessions[mouse_id] = []
            
            # Add this session to the mouse's list
            mouse_sessions[mouse_id].append(session_id)
    
    # Process each session
    for session_id in session_ids:
        try:
            # Load the behaviour data for this session
            data_dict = get_behaviour_data(session_id, cohort)
            
            # Extract the logs and clean the ANSI color codes
            logs = data_dict.get("Logs", [])
            clean_logs = [re.sub(r'\u001b\[\d+m', '', log) for log in logs]
            
            # Prepare the DataFrame structure
            trials_data = []
            
            # Variables to track the current trial
            current_trial = {}
            trial_number = 0
            
            # Parse the logs
            i = 0
            while i < len(clean_logs):
                log_entry = clean_logs[i].split(';')
                
                # Check if this is the start of a new trial (OUT message)
                if log_entry[0] == "OUT" and (i == 0 or (i > 0 and ("C;" in clean_logs[i-1] or i-1 < 0))):
                    # If we have a previous trial, add it to our data
                    if current_trial and 'start_time' in current_trial:
                        trials_data.append(current_trial)
                    
                    # Start a new trial
                    trial_number += 1
                    current_trial = {
                        'trial_number': trial_number,
                        'start_time': float(log_entry[1]),
                        'initial_port': None,
                        'cue_port': None,
                        'outcome': 'Unknown'
                    }
                
                # Check if this is an initial response (R message after OUT)
                elif log_entry[0] == "IN" and len(log_entry) >= 5 and log_entry[2] == "R" and 'start_time' in current_trial and current_trial['initial_port'] is None:
                    try:
                        # Handle the case where the port might be special values
                        port_value = log_entry[3]
                        if port_value == 'F':
                            current_trial['initial_port'] = 'F'
                        else:
                            current_trial['initial_port'] = int(port_value)
                    except ValueError:
                        # If conversion fails, keep as string
                        current_trial['initial_port'] = log_entry[3]
                
                # Check if this is a cue response (C message)
                elif log_entry[0] == "IN" and len(log_entry) >= 5 and log_entry[2] == "C":
                    try:
                        # Handle the case where the port might be special values
                        port_value = log_entry[3]
                        if port_value == 'F':
                            current_trial['cue_port'] = 'F'
                            current_trial['outcome'] = 'Timeout'
                        else:
                            current_trial['cue_port'] = int(port_value)
                    except ValueError:
                        # If conversion fails, keep as string
                        current_trial['cue_port'] = log_entry[3]
                    
                    # Determine outcome
                    if log_entry[4] == "T":
                        current_trial['outcome'] = 'Success'
                    elif log_entry[4] == "F":
                        if current_trial['cue_port'] == 'F':
                            current_trial['outcome'] = 'Timeout'
                        else:
                            current_trial['outcome'] = 'Failure'
                
                i += 1
            
            # Add the last trial if there is one
            if current_trial and 'start_time' in current_trial:
                trials_data.append(current_trial)
            
            # Create the DataFrame
            df = pd.DataFrame(trials_data)
            
            # Count the outcomes
            total_trials = len(df) if not df.empty else 0
            successes = len(df[df['outcome'] == 'Success']) if not df.empty else 0
            failures = len(df[df['outcome'] == 'Failure']) if not df.empty else 0
            
            # Calculate the proportion of attempts (successes + failures) to total trials
            attempts = successes + failures
            attempt_proportion = attempts / total_trials if total_trials > 0 else 0.0
            
            # Store the result for this session
            session_results[session_id] = attempt_proportion
            
        except Exception as e:
            # Handle any errors (file not found, malformed data, etc.)
            print(f"Error processing session {session_id}: {str(e)}")
            session_results[session_id] = 0.0
    
    # Calculate average proportion for each mouse
    mouse_results = {}
    for mouse_id, mouse_session_ids in mouse_sessions.items():
        # Get the proportion values for this mouse's sessions
        mouse_proportions = [session_results.get(session_id, 0.0) for session_id in mouse_session_ids]
        
        # Calculate the average proportion for this mouse
        mouse_average = sum(mouse_proportions) / len(mouse_proportions) if mouse_proportions else 0.0
        
        # Store the result for this mouse
        mouse_results[mouse_id] = mouse_average
    
    # Calculate the overall average across all mice
    average_proportion = sum(mouse_results.values()) / len(mouse_results) if mouse_results else 0.0
    
    # Return both individual mouse results and the overall average
    return {
        'average': average_proportion,
        'mice': mouse_results
    }



"""
Ok so the end thing I want to get is for each day what was the proportion of timeouts per session over the mice in the group. 
This value can either be relative to the proportion of timeouts in the ctrol session before or to the baseline sessions done before the experiment.
I then plot this across the conditions.
So, at first I want to have a function that takes a session and returns the proportion of timeouts per session. This is basically done already above. 
Then the next thing would be taking a list of sessions and finding the average of those values across the sessions. 
One I have the average for a condition I can compare that to different conditions, so I need a function that takes a list of session lists, 
and returns a list containing the timeout proportions of the conditions. This can then be plotted. 
"""