import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

def mouse_heading_cue_offset(sessions,
                             cue_times, 
                             title='title', 
                             start_angle='all',
                             angle_mode='normal',
                             offset=0,
                             cue_mode='both',
                             plot_individual_mice=False):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot
    - plot_type: default: 'line', or 'radial'
    - angle_mode: default: 'normal', or 'delta'
    - offset: default: 0, or a value to add to the cue offset time to get the time to get the heading angle at.
    - plot_individual_mice: whether to plot each mouse's data individually
    """
    
    # Define a color map
    viridis = plt.cm.get_cmap('viridis')
    predefined_colors = {
        'unlimited': viridis(0.0),  
        '1000ms': viridis(0.1),    
        '750ms': viridis(0.2),      
        '500ms': viridis(0.4),
        '300ms': viridis(0.5),      
        '100ms': viridis(0.6),
        '50ms': viridis(0.7),
        '25ms': viridis(0.8),
        '5ms': viridis(1.0)
    }

    def get_trials(session_list):
        trials = {}
        total_trials = []
        for session in session_list:
            mouse = session.session_dict['mouse_id']
            if mouse =='wtjx261-2b' or mouse == 'wtjx261-2a':
                continue
            if mouse not in trials:
                trials[mouse] = {'trials': []}
            if cue_mode == 'both':
                for trial in session.trials:
                    trial['session_object'] = session
                    trials[mouse]['trials'].append(trial)
                    total_trials.append(trial)
            elif cue_mode == 'visual':
                for trial in session.trials:
                    if 'audio' not in trial['correct_port']:
                        trial['session_object'] = session
                        trials[mouse]['trials'].append(trial)
                        total_trials.append(trial)
            elif cue_mode == 'audio':
                for trial in session.trials:
                    if 'audio' in trial['correct_port']:
                        trial['session_object'] = session
                        trials[mouse]['trials'].append(trial)
                        total_trials.append(trial)
        return total_trials, trials

    data_sets = {}
    for session_list, cue_time in zip(sessions, cue_times):
        total_trials, trials = get_trials(session_list)
        if cue_time not in data_sets:
            data_sets[cue_time] = {'total_trials': [], 'trials': defaultdict(list)}

        data_sets[cue_time]['total_trials'].extend(total_trials)
        for mouse, trial_list in trials.items():
            data_sets[cue_time]['trials'][mouse].extend(trial_list['trials'])

    for cue_time, data in data_sets.items():
        data_sets[cue_time]['trials'] = {mouse: {'trials': trial_list} for mouse, trial_list in data['trials'].items()}

    # Sort trials into successful and unsuccessful trials:
    for cue_time in data_sets:
        trials = data_sets[cue_time]['trials']
        for mouse in trials:
            successful_trials = []
            unsuccessful_trials = []
            timeouts = []
            for trial in trials[mouse]['trials']:
                if trial["next_sensor"] != {}:
                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                        successful_trials.append(trial)
                    else:
                        unsuccessful_trials.append(trial)
                else:
                    timeouts.append(trial)

            data_sets[cue_time]['trials'][mouse]['successful_trials'] = successful_trials
            data_sets[cue_time]['trials'][mouse]['unsuccessful_trials'] = unsuccessful_trials
            data_sets[cue_time]['trials'][mouse]['timeouts'] = timeouts

    def get_data(trials, mouse_id):
        percentages = []
        for trial in trials:
            session = trial['session_object']
            dlc_data = trial.get('DLC_data')
            if len(trial['video_frames']) > 0:
                cue_offset_time = trial['cue_end'] + offset
                start_angle = trial['turn_data']['cue_presentation_angle']

                mouse_start_bearing = trial['turn_data']['bearing']

                timestamps = dlc_data['timestamps']
                index = np.searchsorted(timestamps, cue_offset_time, side='left') - 1
                cue_offset_coords = dlc_data.iloc[index]

                left_ear_coords = (cue_offset_coords["left_ear"]["x"], cue_offset_coords["left_ear"]["y"])
                right_ear_coords = (cue_offset_coords["right_ear"]["x"], cue_offset_coords["right_ear"]["y"])

                vector_x = right_ear_coords[0] - left_ear_coords[0]
                vector_y = right_ear_coords[1] - left_ear_coords[1]

                theta_rad = math.atan2(-vector_y, vector_x)
                theta_deg = (math.degrees(theta_rad) + 90) % 360

                midpoint_x = (left_ear_coords[0] + right_ear_coords[0]) / 2
                midpoint_y = (left_ear_coords[1] + right_ear_coords[1]) / 2

                port_coordinates = session.port_coordinates
                cue_offset_relative_angles = []
                mouse_heading_rad = np.deg2rad(theta_deg)

                for port_x, port_y in port_coordinates:
                    vector_x = port_x - midpoint_x
                    vector_y = port_y - midpoint_y
                    port_angle_rad = math.atan2(-vector_y, vector_x)
                    relative_angle_deg = math.degrees(port_angle_rad - mouse_heading_rad) % 360
                    cue_offset_relative_angles.append(relative_angle_deg)

                correct_port = trial["correct_port"]
                if correct_port == "audio-1":
                    correct_port = 1
                port = int(correct_port) - 1
                cue_angle = cue_offset_relative_angles[port] % 360
                if cue_angle > 180:
                    cue_angle -= 360
                elif cue_angle <= -180:
                    cue_angle += 360

                end_angle = cue_angle
                delta_angle = start_angle - end_angle % 360

                if delta_angle > 180:
                    delta_angle -= 360
                elif delta_angle <= -180:
                    delta_angle += 360

                percentage_turn = delta_angle / start_angle
                percentages.append(percentage_turn)

        average_percentage = np.mean(percentages)
        sem = np.std(percentages) / np.sqrt(len(percentages))
        sd = np.std(percentages)

        return average_percentage

    success_data = {}
    unsuccessful_data = {}

    for cue_time in data_sets:
        if cue_time not in success_data:
            success_data[cue_time] = {'data': {}, 'sem': {}, 'sd': {}}
            unsuccessful_data[cue_time] = {'data': {}, 'sem': {}, 'sd': {}}

        trials = data_sets[cue_time]['trials']
        for mouse in trials:
            successful_trials = trials[mouse]['successful_trials']
            unsuccessful_trials = trials[mouse]['unsuccessful_trials']

            data = get_data(successful_trials, mouse)
            if mouse not in success_data[cue_time]['data']:
                success_data[cue_time]['data'][mouse] = []
            success_data[cue_time]['data'][mouse].append(data)

            data = get_data(unsuccessful_trials, mouse)
            if mouse not in unsuccessful_data[cue_time]['data']:
                unsuccessful_data[cue_time]['data'][mouse] = []
            unsuccessful_data[cue_time]['data'][mouse].append(data)
            unsuccessful_data[cue_time]['data'][mouse] = np.mean(unsuccessful_data[cue_time]['data'][mouse])

    for cue_time in success_data:
        data = []
        for mouse in success_data[cue_time]['data']:
            data.append(success_data[cue_time]['data'][mouse])
        success_data[cue_time]['array'] = np.array(data)
        success_data[cue_time]['mean'] = np.mean(data)
        success_data[cue_time]['sem'] = np.std(data) / np.sqrt(len(data))
        success_data[cue_time]['sd'] = np.std(data)

        data = []
        for mouse in unsuccessful_data[cue_time]['data']:
            data.append(unsuccessful_data[cue_time]['data'][mouse])
        unsuccessful_data[cue_time]['array'] = np.array(data)
        unsuccessful_data[cue_time]['mean'] = np.mean(data)
        unsuccessful_data[cue_time]['sem'] = np.std(data) / np.sqrt(len(data))
        unsuccessful_data[cue_time]['sd'] = np.std(data)

    # Plotting the data:
    bin_titles = [cue_time for cue_time in predefined_colors.keys() if cue_time in success_data]
    averages = [success_data[title]['mean'] for title in bin_titles]
    sems = [success_data[title]['sem'] for title in bin_titles]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bin_titles))  # X-axis positions (one for each cue time)

    if plot_individual_mice:
        # Step 1: Build a dictionary with mouse data for each cue time
        mouse_data_dict = {}

        for cue_time in success_data:
            for mouse in success_data[cue_time]['data']:
                if mouse not in mouse_data_dict:
                    mouse_data_dict[mouse] = [np.nan] * len(bin_titles)  # Initialize with NaN values for each cue time

                for idx, title in enumerate(bin_titles):
                    # Check if the mouse has data for the specific cue_time and if the value is not already filled
                    if mouse in success_data[title]['data'] and success_data[title]['data'][mouse] and np.isnan(mouse_data_dict[mouse][idx]):
                        # Extract the value from the list and assign it to the correct position
                        mouse_data_dict[mouse][idx] = success_data[title]['data'][mouse][0]

        # Step 2: Plot the data from the dictionary
        for mouse, mouse_data in mouse_data_dict.items():
            ax.plot(x, mouse_data, label=f'Mouse {mouse}', linestyle='--', marker='o')

        # Optionally, add a legend to identify each mouse
        ax.legend(loc='upper right')

    # Plot the averages as a solid line
    ax.plot(x, averages, color='b', marker='o', linestyle='-', lw=2)

    # Shading the area to represent SEM
    ax.fill_between(x, 
                    np.array(averages) - np.array(sems),  # Lower bound (mean - SEM)
                    np.array(averages) + np.array(sems),  # Upper bound (mean + SEM)
                    color='b', alpha=0.2)

    # Customizing the plot
    ax.set_xticks(x)
    ax.set_xticklabels(bin_titles, rotation=45, ha="right")
    ax.set_xlabel('Cue presentation duration (ms)', fontsize=14)
    ax.set_ylabel('% of turn completed at cue offset', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
