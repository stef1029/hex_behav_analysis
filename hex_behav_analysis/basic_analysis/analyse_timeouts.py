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
    session_dict=cohort.get_session(test_session)
    file_path = Path(session_dict.get("raw_data").get("behaviour_data"))
    if file_path is not None and file_path.exists():
        with file_path.open("r") as f:
            bd = json.load(f)

    return bd

import pandas as pd
import re
from datetime import datetime

def parse_mouse_trials(data_dict):
    """
    Parse a dictionary containing mouse trial data and return counts of different outcomes.
    
    Args:
        data_dict (dict): Dictionary containing mouse trial data with logs
        
    Returns:
        tuple: (successes, failures, timeouts) counts from the trials
    """
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
    successes = len(df[df['outcome'] == 'Success']) if not df.empty else 0
    failures = len(df[df['outcome'] == 'Failure']) if not df.empty else 0
    timeouts = len(df[df['outcome'] == 'Timeout']) if not df.empty else 0
    
    return (successes, failures, timeouts)
    
    return df

def calculate_success_percentage(df):
    """
    Calculate the percentage of successful trials from the trials DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the parsed trial data
        
    Returns:
        float: The success percentage (0-100)
    """
    if df.empty:
        return 0.0
    
    # Count the number of successful trials
    success_count = len(df[df['outcome'] == 'Success'])
    
    # Calculate the percentage
    success_percentage = (success_count / len(df)) * 100
    
    return success_percentage

if __name__ == "__main__":

    cohort_directory = "/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE/Experiment"

    cohort = Cohort_folder(cohort_directory, multi=True, OEAB_legacy=False, use_existing_cohort_info=True)

    test_session = "250130_170712_wtjp271-5b"

    bd = get_behaviour_data(test_session, cohort)

    data = parse_mouse_trials(bd)
    print("Successes: ", data[0])
    print("Failures: ", data[1])        
    print("Timeouts: ", data[2])