from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
import json
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from colorama import Fore, Style, init

init()

def calculate_total_time_seconds(experiment_data):
    """
    Calculate the total time in seconds between the start and end times in the experiment data.
    
    Args:
        experiment_data (dict): Dictionary containing experiment metadata with 'Date and time' 
                               and 'End time' fields.
    
    Returns:
        float: Total time in seconds.
    
    Notes:
        - 'Date and time' format is expected to be 'YYMMDD_HHMMSS'
        - 'End time' format is expected to be 'YYMMDDHHMMSS'
    """
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

def get_metadata(session_id, cohort):
    """
    Extract metadata for a given session including duration, trial count, and cue duration.
    
    Args:
        session_id (str): Unique identifier for the session
        cohort (Cohort_folder): The cohort object containing session data
    
    Returns:
        dict: Dictionary containing total_time, num_trials, and cue_duration
    """
    session_dict = cohort.get_session(session_id)
    if session_dict is None:
        print(f"Session {session_id} not found in cohort.")
        return None
    
    behaviour_data_metadata = session_dict.get("raw_data", {}).get("behaviour_data", {})
    with open(behaviour_data_metadata, 'r') as f:
        metadata = json.load(f)

    total_time = calculate_total_time_seconds(metadata)
    num_trials = metadata.get("Total trials", None)
    cue_duration = metadata.get("Cue duration", None)

    return {"total_time": total_time, "num_trials": num_trials, "cue_duration": cue_duration}

def main():
    """
    Main function to discover and organise experimental sessions.
    
    Set print_sessions = True to discover sessions that match your criteria.
    Set print_sessions = False to return organised session groups with manually curated lists.
    """
    
    # CONFIGURATION
    print_sessions = False  # Set to True for discovery mode, False for data return mode
    
    # UPDATE THESE PATHS AND CRITERIA FOR YOUR EXPERIMENT
    cohort_directory = "/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment"  # Update this path
    
    # Define your experimental conditions with dates only
    # The script will automatically find all mice that have sessions on these dates
    saline_d1 = {"name": "Saline D1", "dates": ["250516"]}
    saline_d2 = {"name": "Saline D2", "dates": ["250521"]}
    saline_d3 = {"name": "Saline D3", "dates": ["250527"]}
    saline_d4 = {"name": "Saline D4", "dates": ["250603"]}
    saline_d5 = {"name": "Saline D5", "dates": ["250610"]}
    
    psem_d1 = {"name": "PSEM D1", "dates": ["250519"]}
    psem_d2 = {"name": "PSEM D2", "dates": ["250520"]}
    psem_d3 = {"name": "PSEM D3", "dates": ["250602"]}
    psem_d4 = {"name": "PSEM D4", "dates": ["250604"]}
    psem_d5 = {"name": "PSEM D5", "dates": ["250611"]}

    dcz_d1 = {"name": "DCZ D1", "dates": ["250522"]}
    dcz_d2 = {"name": "DCZ D2", "dates": ["250528"]}
    dcz_d3 = {"name": "DCZ D3", "dates": ["250529"]}
    dcz_d4 = {"name": "DCZ D4", "dates": ["250530"]}
    dcz_d5 = {"name": "DCZ D5", "dates": ["250609"]}
    
    # Compile all conditions into a list
    session_groups = [saline_d1, saline_d2, saline_d3, saline_d4, saline_d5,
                      psem_d1, psem_d2, psem_d3, psem_d4, psem_d5,
                      dcz_d1, dcz_d2, dcz_d3, dcz_d4, dcz_d5]
    
    # Convert to dictionary for easier access
    session_groups = {group["name"]: group for group in session_groups}

    # DISCOVERY MODE - Print sessions to help identify which ones to keep
    if print_sessions:
        print("Running in discovery mode...")
        print("Sessions matching your criteria will be displayed with metadata.")
        print("Use this information to manually curate your session lists below.\n")
        
        cohort = Cohort_folder(cohort_directory, 
                               multi=True, 
                               OEAB_legacy=False, 
                               use_existing_cohort_info=False,
                               ignore_tests=True)
        
        phases = cohort.phases()
        
        # Define colours for different mice
        colours = [Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.RED, Fore.GREEN, Fore.YELLOW]
        
        for session_group in session_groups:
            name = session_groups[session_group]["name"]
            dates = session_groups[session_group]["dates"]
            print(f"\n{Fore.WHITE}{Style.BRIGHT}{name}{Style.RESET_ALL}")
            
            # Find all mice that have sessions on these dates
            mice_on_dates = set()
            for phase in phases:
                for session in phases[phase]:
                    date = str(session[:6])
                    if date in dates:
                        mouse = phases[phase][session]["mouse"]
                        mice_on_dates.add(mouse)
            
            mice_on_dates = sorted(list(mice_on_dates))  # Sort for consistent ordering
            
            # Create mouse to colour mapping
            mouse_colour_map = {}
            for i, mouse in enumerate(mice_on_dates):
                mouse_colour_map[mouse] = colours[i % len(colours)]
            
            # Store mice in the session group for later use
            session_groups[session_group]["mice"] = mice_on_dates
            
            dates_seen = []
            for phase in phases:
                for session in phases[phase]:
                    mouse = phases[phase][session]["mouse"]
                    date = str(session[:6])
                    if mouse in mice_on_dates and date in dates:
                        if date + mouse not in dates_seen:
                            dates_seen.append(date + mouse)
                            print("")
                        
                        mouse_id = phases[phase][session]["mouse"]
                        colour = mouse_colour_map[mouse_id]
                        
                        # Get metadata for this session
                        metadata = get_metadata(session, cohort)
                        
                        # Print session with appropriate colour
                        print(f"{colour}Session: {session}: {metadata}{Style.RESET_ALL}")

    # MANUAL SESSION CURATION
    # After running discovery mode, manually add your selected sessions here
    # Each condition should have 'sessions_ctrl' and 'sessions_test' lists
    
    # Example structure - replace with your actual session IDs after discovery
    saline_d1["sessions"] = ['250516_154445_mtao108-3e',
                            '250516_130123_mtao101-3c',
                            '250516_142239_mtao101-3g',
                            '250516_142250_mtao106-3b',
                            '250516_154435_mtao106-1e',
                            '250516_171208_mtao102-3e',
                            '250516_171157_mtao102-3c',
                            '250516_114848_mtao107-2a',
                            '250516_130116_mtao101-3b',
                            '250516_114838_mtao106-3a']
    
    saline_d2["sessions"] = ['250521_150135_mtao108-3e',
                            '250521_105428_mtao101-3c',
                            '250521_133738_mtao101-3g',
                            '250521_133747_mtao106-3b',
                            '250521_150148_mtao106-1e',
                            '250521_121720_mtao102-3e',
                            '250521_121707_mtao102-3c',
                            '250521_093522_mtao107-2a',
                            '250521_105421_mtao101-3b',
                            '250521_093516_mtao106-3a']

    saline_d3["sessions"] = ['250527_132422_mtao101-3c',
                            '250527_120245_mtao106-3b',
                            '250527_132414_mtao106-1e',
                            '250527_164046_mtao102-3e',
                            '250527_161957_mtao102-3e',
                            '250527_145343_mtao102-3c',
                            '250527_152525_mtao102-3c',
                            '250527_120252_mtao107-2a',
                            '250527_145353_mtao101-3b',
                            '250527_152531_mtao101-3b'
                            ]

    saline_d4["sessions"] = ['250603_133634_mtao101-3c',
                            '250603_124457_mtao102-3e',
                            '250603_122058_mtao102-3e',
                            '250603_124505_mtao102-3c',
                            '250603_122109_mtao102-3c',
                            '250603_110309_mtao106-3b',
                            '250603_110319_mtao106-1e',
                            '250603_133625_mtao107-2a']

    saline_d5["sessions"] = ['250610_151304_mtao101-3c',
                            '250610_122809_mtao106-3b',
                            '250610_122819_mtao106-1e',
                            '250610_134837_mtao102-3e',
                            '250610_142448_mtao102-3e',
                            '250610_134848_mtao102-3c',
                            '250610_142456_mtao102-3c',
                            '250610_151310_mtao107-2a']

    psem_d1["sessions"] = ['250519_164007_mtao108-3e',
                            '250519_120521_mtao101-3c',
                            '250519_150125_mtao101-3g',
                            '250519_150138_mtao106-3b',
                            '250519_163953_mtao106-1e',
                            '250519_132004_mtao102-3e',
                            '250519_131950_mtao102-3c',
                            '250519_100032_mtao107-2a',
                            '250519_120515_mtao101-3b',
                            '250519_100024_mtao106-3a']

    psem_d2["sessions"] = ['250520_143937_mtao108-3e',
                            '250520_101738_mtao101-3c',
                            '250520_131404_mtao101-3g',
                            '250520_131417_mtao106-3b',
                            '250520_143949_mtao106-1e',
                            '250520_114457_mtao102-3e',
                            '250520_114416_mtao102-3c',
                            '250520_083745_mtao107-2a',
                            '250520_101729_mtao101-3b',
                            '250520_083738_mtao106-3a']

    psem_d3["sessions"] = ['250602_125622_mtao108-3e',
                            '250602_125614_mtao101-3c',
                            '250602_095944_mtao106-3b',
                            '250602_095933_mtao106-1e',
                            '250602_113012_mtao102-3e',
                            '250602_113023_mtao102-3c',
                            '250602_100251_mtao106-3b',
                            '250602_100302_mtao106-1e']

    psem_d4["sessions"] = ['250604_152850_mtao108-3e',
                            '250604_134134_mtao101-3g',
                            '250604_134123_mtao106-3b',
                            '250604_152840_mtao106-1e',
                            '250604_121620_mtao102-3e',
                            '250604_121608_mtao102-3c']

    psem_d5["sessions"] = ['250611_141617_mtao101-3c',
                            '250611_112216_mtao106-3b',
                            '250611_112225_mtao106-1e',
                            '250611_125127_mtao102-3e',
                            '250611_125137_mtao102-3c',
                            '250611_141626_mtao107-2a']

    dcz_d1["sessions"] = ['250522_160945_mtao108-3e',
                            '250522_163100_mtao108-3e',
                            '250522_134203_mtao101-3g',
                            '250522_134148_mtao106-3b',
                            '250522_160928_mtao106-1e',
                            '250522_163114_mtao106-1e',
                            '250522_120444_mtao102-3e',
                            '250522_120505_mtao102-3c']

    dcz_d2["sessions"] = ['250528_125157_mtao101-3c',
                            '250528_111501_mtao106-3b',
                            '250528_114220_mtao106-3b',
                            '250528_125147_mtao106-1e',
                            '250528_161041_mtao102-3e',
                            '250528_152847_mtao102-3e',
                            '250528_141345_mtao102-3c',
                            '250528_111507_mtao107-2a',
                            '250528_114232_mtao107-2a',
                            '250528_141354_mtao101-3b']

    dcz_d3["sessions"] = ['250529_143840_mtao101-3c',
                            '250529_130834_mtao106-3b',
                            '250529_143851_mtao106-1e',
                            '250529_155722_mtao102-3e',
                            '250529_155737_mtao102-3c',
                            '250529_130822_mtao107-2a']

    dcz_d4["sessions"] = ['250530_152902_mtao101-3c',
                            '250530_150449_mtao101-3c',
                            '250530_120955_mtao106-3b',
                            '250530_120946_mtao106-1e',
                            '250530_141813_mtao102-3e',
                            '250530_133605_mtao102-3e',
                            '250530_141824_mtao102-3c',
                            '250530_133616_mtao102-3c',
                            '250530_152855_mtao107-2a',
                            '250530_150443_mtao107-2a']

    dcz_d5["sessions"] = ['250609_140800_mtao101-3c',
                            '250609_132022_mtao102-3e',
                            '250609_124408_mtao102-3c',
                            '250609_132033_mtao102-3c',
                            '250609_121034_mtao106-3b',
                            '250609_111910_mtao106-3b',
                            '250609_121043_mtao106-1e',
                            '250609_111921_mtao106-1e',
                            '250609_140808_mtao107-2a']
    
    
    # DATA RETURN MODE - Return organised session groups
    if not print_sessions:
        print(f"Loaded {len(session_groups)} session groups")
        print("Structure of dictionaries: dict_keys(['name', 'dates', 'mice', 'sessions_ctrl', 'sessions_test'])")
        print("Session group names:")
        for group in session_groups.keys():
            print(f"  - {group}")
        
        return session_groups

if __name__ == "__main__":
    main()