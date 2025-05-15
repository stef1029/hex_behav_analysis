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
    # Extract the start time string
    start_time_str = experiment_data["Date and time"]
    
    # Extract the end time string
    end_time_str = experiment_data["End time"]
    
    # Parse the start time
    # Format: YYMMDD_HHMMSS
    start_year = int("20" + start_time_str[0:2])
    start_month = int(start_time_str[2:4])
    start_day = int(start_time_str[4:6])
    # Skip the underscore
    start_hour = int(start_time_str[7:9])
    start_minute = int(start_time_str[9:11])
    start_second = int(start_time_str[11:13])
    
    # Parse the end time
    # Format: YYMMDDHHMMSS
    end_year = int("20" + end_time_str[0:2])
    end_month = int(end_time_str[2:4])
    end_day = int(end_time_str[4:6])
    end_hour = int(end_time_str[6:8])
    end_minute = int(end_time_str[8:10])
    end_second = int(end_time_str[10:12])
    
    # Calculate total seconds
    from datetime import datetime
    
    start_datetime = datetime(
        start_year, start_month, start_day, 
        start_hour, start_minute, start_second
    )
    
    end_datetime = datetime(
        end_year, end_month, end_day,
        end_hour, end_minute, end_second
    )
    
    # Calculate the time difference in seconds
    time_difference = (end_datetime - start_datetime).total_seconds()
    
    return time_difference

def get_metadata(session_id, cohort):
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
    print_sessions = False

    group_1 = ["wtjp280-4a", "wtjp271-5b", "wtjp271-5c"]
    group_2 = ["wtjp280-4b", "wtjp271-5d", "wtjp280-4f"]

    # group_1_baseline_sessions = {"name": "Group 1 Baseline", "dates": ["250127", "250128", "250129", "250130", "250131", "250203", "250205"], "mice": group_1}
    # group_2_baseline_sessions = {"name": "Group 2 Baseline", "dates": ["250127", "250128", "250129", "250130", "250131", "250203", "250205"], "mice": group_2}
    group_1_saline = {"name": "Group 1 saline", "dates": ["250206", "250207"], "mice": group_1}
    # group_1_1ng = {"name": "Group 1 1ng", "dates": ["250219", "250220"], "mice": group_1}
    group_1_10ng = {"name": "Group 1 10ng", "dates": ["250208", "250209"], "mice": group_1}
    group_1_100ng = {"name": "Group 1 100ng", "dates": ["250210", "250211"], "mice": group_1}
    group_1_500ng = {"name": "Group 1 500ng", "dates": ["250212", "250213"], "mice": group_1}
    group_1_1000ng = {"name": "Group 1 1000ng", "dates": ["250214", "250215"], "mice": group_1}
    group_2_saline = {"name": "Group 2 saline", "dates": ["250214", "250215"], "mice": group_2}
    # group_2_1ng = {"name": "Group 2 1ng", "dates": ["250219", "250220"], "mice": group_2}
    group_2_10ng = {"name": "Group 2 10ng", "dates": ["250212", "250213"], "mice": group_2}
    group_2_100ng = {"name": "Group 2 100ng", "dates": ["250210", "250211"], "mice": group_2}
    group_2_500ng = {"name": "Group 2 500ng", "dates": ["250208", "250209"], "mice": group_2}
    group_2_1000ng = {"name": "Group 2 1000ng", "dates": ["250206", "250207"], "mice": group_2}

    # session_groups = [group_1_baseline_sessions, group_2_baseline_sessions,
    #                 group_1_saline, group_1_1ng, group_1_10ng, group_1_100ng, group_1_500ng, group_1_1000ng,
    #                 group_2_saline, group_2_1ng, group_2_10ng, group_2_100ng, group_2_500ng, group_2_1000ng]

    session_groups = [group_1_saline, group_1_10ng, group_1_100ng, group_1_500ng, group_1_1000ng,
                    group_2_saline, group_2_10ng, group_2_100ng, group_2_500ng, group_2_1000ng]
    
    session_groups = {group["name"]: group for group in session_groups}

    if print_sessions:

        cohort_directory = "/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE/Experiment"

        cohort = Cohort_folder(cohort_directory, multi=True, OEAB_legacy=False, use_existing_cohort_info=True)

        """
        for session list i want to make, i need to filter all the sessions by the mice in the group and the dates that the session happened on. 
        I think need to print all the sessions along with their session lengths, in time order and filter to the ones I want to use. 

        14 groups total
        """

        phases = cohort.phases()

        print("Cohort phases dictionary:")
        # print(json.dumps(phases, indent=4))

        # Define a list of colors to cycle through
        colors = [Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.RED, Fore.GREEN, Fore.YELLOW]

        for session_group in session_groups:
            name = session_groups[session_group]["name"]
            mice = session_groups[session_group]["mice"]
            dates = session_groups[session_group]["dates"]
            print(f"\n{Fore.WHITE}{Style.BRIGHT}{name}{Style.RESET_ALL}")
            
            # Create a dictionary to track mouse to color mapping
            mouse_color_map = {}
            for i, mouse in enumerate(mice):
                # Assign a color to each mouse, cycling through the colors list
                mouse_color_map[mouse] = colors[i % len(colors)]
            dates_seen = []
            for phase in phases:
                for session in phases[phase]:
                    # Check if the session is in the current group
                    mouse = phases[phase][session]["mouse"]
                    date = str(session[:6])
                    if mouse in mice and date in dates:
                        if date+mouse not in dates_seen:
                            dates_seen.append(date+mouse)
                            print("")
                        mouse_id = phases[phase][session]["mouse"]
                        color = mouse_color_map[mouse_id]
                        
                        # Get metadata for this session
                        metadata = get_metadata(session, cohort)
                        
                        # Print with the appropriate color
                        print(f"{color}Session: {session}: {metadata}{Style.RESET_ALL}")

    # group_1_baseline_sessions["sessions_ctrl"] = ["250127_181421_wtjp271-5b", "250128_113115_wtjp271-5b", "250129_191702_wtjp271-5b", "250130_170712_wtjp271-5b", "250131_183201_wtjp271-5b", "250203_175639_wtjp271-5b", "250205_173227_wtjp271-5b",
    #                                         "250127_185021_wtjp271-5c", "250128_113116_wtjp271-5c", "250129_191702_wtjp271-5c", "250130_170712_wtjp271-5c", "250131_183201_wtjp271-5c", "250203_175639_wtjp271-5c", "250205_173227_wtjp271-5c",
    #                                         "250127_145632_wtjp280-4a", "250128_082732_wtjp280-4a", "250129_164140_wtjp280-4a", "250130_142223_wtjp280-4a", "250131_162811_wtjp280-4a", "250203_160202_wtjp280-4a", "250205_155943_wtjp280-4a"]

    # group_2_baseline_sessions["sessions_ctrl"] = ["250127_173355_wtjp271-5d", "250128_100504_wtjp271-5d", "250129_175317_wtjp271-5d", "250130_161447_wtjp271-5d", "250131_173050_wtjp271-5d", "250203_165929_wtjp271-5d", "250205_190259_wtjp271-5d",
    #                                             "250127_154906_wtjp280-4b", "250128_082732_wtjp280-4b", "250129_164140_wtjp280-4b", "250130_142223_wtjp280-4b", "250131_162810_wtjp280-4b", "250203_160202_wtjp280-4b", "250205_155943_wtjp280-4b",
    #                                             "250127_163919_wtjp280-4f", "250128_110948_wtjp280-4f", "250129_184500_wtjp280-4f", "250130_161448_wtjp280-4f", "250131_173050_wtjp280-4f", "250203_165929_wtjp280-4f", "250205_190300_wtjp280-4f"]


    group_1_saline["sessions_ctrl"] = ["250206_163413_wtjp271-5b", "250206_165447_wtjp271-5b", "250207_124249_wtjp271-5b", "250207_130006_wtjp271-5b",
                                        "250206_165448_wtjp271-5c", "250206_163413_wtjp271-5c", "250207_130006_wtjp271-5c",
                                        "250206_155341_wtjp280-4a", "250207_120337_wtjp280-4a"]

    group_1_saline["sessions_test"] = ["250206_195810_wtjp271-5b", "250207_161602_wtjp271-5b", "250207_173538_wtjp271-5b",
                                        "250206_195811_wtjp271-5c", "250207_161602_wtjp271-5c", "250207_173538_wtjp271-5c",
                                        "250206_181915_wtjp280-4a", "250207_141359_wtjp280-4a"]


    group_1_10ng["sessions_ctrl"] = ["250208_134802_wtjp271-5b", "250209_120731_wtjp271-5b",
                                        "250208_134803_wtjp271-5c", "250209_120731_wtjp271-5c",
                                        "250208_125823_wtjp280-4a", "250209_112807_wtjp280-4a",]

    group_1_10ng["sessions_test"] = ["250208_172218_wtjp271-5b", "250209_152119_wtjp271-5b",
                                        "250208_172219_wtjp271-5c", "250209_152120_wtjp271-5c",
                                        "250208_153047_wtjp280-4a", "250209_133023_wtjp280-4a"]


    group_1_100ng["sessions_ctrl"] = ["250210_113704_wtjp271-5b",
                                        "250211_115517_wtjp271-5b",
                                        "250210_113705_wtjp271-5c",
                                        "250211_115518_wtjp271-5c",
                                        "250210_105724_wtjp280-4a",
                                        "250211_110617_wtjp280-4a"]


    group_1_100ng["sessions_test"] = ["250210_151858_wtjp271-5b",
                                        "250211_152549_wtjp271-5b",
                                        "250210_151859_wtjp271-5c",
                                        "250211_152549_wtjp271-5c",
                                        "250210_131640_wtjp280-4a",
                                        "250211_133312_wtjp280-4a"]


    group_1_500ng["sessions_ctrl"] = ["250212_112526_wtjp271-5b",
                                        "250213_120012_wtjp271-5b",
                                        "250212_112526_wtjp271-5c",
                                        "250213_120012_wtjp271-5c",
                                        "250212_104121_wtjp280-4a",
                                        "250213_111628_wtjp280-4a"]

    group_1_500ng["sessions_test"] = ["250212_145702_wtjp271-5b",
                                        "250213_152927_wtjp271-5b",
                                        "250212_145702_wtjp271-5c",
                                        "250213_152928_wtjp271-5c",
                                        "250212_130035_wtjp280-4a",
                                        "250213_133506_wtjp280-4a"]



    group_1_1000ng["sessions_ctrl"] = ["250214_123023_wtjp271-5b",
                                        "250215_112555_wtjp271-5b",
                                        "250214_123024_wtjp271-5c",
                                        "250215_112555_wtjp271-5c",
                                        "250214_114157_wtjp280-4a",
                                        "250215_103705_wtjp280-4a"]

    group_1_1000ng["sessions_test"] = ["250214_162403_wtjp271-5b",
                                        "250215_150936_wtjp271-5b",
                                        "250214_162404_wtjp271-5c",
                                        "250215_150936_wtjp271-5c",
                                        "250214_140845_wtjp280-4a",
                                        "250214_151515_wtjp280-4a",
                                        "250215_131410_wtjp280-4a"]



    group_2_saline["sessions_ctrl"] = ["250214_132226_wtjp271-5d",
                                        "250215_121426_wtjp271-5d",
                                        "250214_114157_wtjp280-4b",
                                        "250215_103705_wtjp280-4b",
                                        "250214_132226_wtjp280-4f",
                                        "250215_121426_wtjp280-4f"]

    group_2_saline["sessions_test"] = ["250214_181541_wtjp271-5d",
                                        "250215_165957_wtjp271-5d",
                                        "250214_140845_wtjp280-4b",
                                        "250214_151515_wtjp280-4b",
                                        "250215_131410_wtjp280-4b",
                                        "250214_181542_wtjp280-4f",
                                        "250215_165957_wtjp280-4f"]


    group_2_10ng["sessions_ctrl"] = ["250212_121236_wtjp271-5d",
                                        "250213_124051_wtjp271-5d",
                                        "250212_104121_wtjp280-4b",
                                        "250213_111629_wtjp280-4b",
                                        "250212_121236_wtjp280-4f",
                                        "250213_124052_wtjp280-4f"]

    group_2_10ng["sessions_test"] = ["250212_164818_wtjp271-5d",
                                        "250213_172052_wtjp271-5d",
                                        "250212_130036_wtjp280-4b",
                                        "250213_133507_wtjp280-4b",
                                        "250212_164818_wtjp280-4f",
                                        "250213_172052_wtjp280-4f"]


    group_2_100ng["sessions_ctrl"] = ["250210_122058_wtjp271-5d",
                                        "250211_123816_wtjp271-5d",
                                        "250210_105724_wtjp280-4b",
                                        "250211_110617_wtjp280-4b",
                                        "250211_113006_wtjp280-4b",
                                        "250210_122058_wtjp280-4f",
                                        "250211_123816_wtjp280-4f"]

    group_2_100ng["sessions_test"] = ["250210_170651_wtjp271-5d",
                                        "250211_171548_wtjp271-5d",
                                        "250210_131641_wtjp280-4b",
                                        "250211_133312_wtjp280-4b",
                                        "250210_170651_wtjp280-4f",
                                        "250211_171549_wtjp280-4f"]


    group_2_500ng["sessions_ctrl"] = ["250208_142953_wtjp271-5d",
                                        "250209_124928_wtjp271-5d",
                                        "250208_125823_wtjp280-4b",
                                        "250209_112807_wtjp280-4b",
                                        "250208_142953_wtjp280-4f",
                                        "250209_124928_wtjp280-4f"]

    group_2_500ng["sessions_test"] = ["250208_191003_wtjp271-5d",
                                        "250209_171209_wtjp271-5d",
                                        "250208_153048_wtjp280-4b",
                                        "250209_133023_wtjp280-4b",
                                        "250208_191003_wtjp280-4f",
                                        "250209_171209_wtjp280-4f"]


    group_2_1000ng["sessions_ctrl"] = ["250206_173850_wtjp271-5d",
                                        "250207_133105_wtjp271-5d",
                                        "250206_155341_wtjp280-4b",
                                        "250207_120337_wtjp280-4b",
                                        "250206_173851_wtjp280-4f",
                                        "250207_133105_wtjp280-4f"]

    group_2_1000ng["sessions_test"] = ["250206_214741_wtjp271-5d",
                                        "250206_224631_wtjp271-5d",
                                        "250206_232054_wtjp271-5d",
                                        "250207_182039_wtjp271-5d",
                                        "250206_181915_wtjp280-4b",
                                        "250207_141359_wtjp280-4b",
                                        "250206_214742_wtjp280-4f",
                                        "250206_224631_wtjp280-4f",
                                        "250206_232054_wtjp280-4f",
                                        "250207_182039_wtjp280-4f"]
    
    print(f"Loaded {len(session_groups)}session groups")
    print(f"Structure of dicts: dict_keys(['name', 'dates', 'mice', 'sessions_ctrl', 'sessions_test'])")
    print("Session group names:")
    for group in session_groups.keys():
        print(group)
    
    return session_groups

if __name__ == "__main__":
    main()