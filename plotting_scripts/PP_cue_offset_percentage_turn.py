import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
from datetime import datetime

def lighten_color(color, factor=0.5):
    """Creates a lighter version of a color for plot shading"""
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def mouse_heading_cue_offset(sessions,
                             cue_times, 
                             plot_title='title',
                             x_label = 'x_label',
                             y_label = 'y_label', 
                             start_angle='all',
                             angle_mode='normal',
                             offset=0,
                             cue_mode='both',
                             plot_individual_mice=False,
                             exclusion_mice = [],
                             output_path=None,
                             plot_save_name = 'untitled_plot',
                             draft=True):
    """
    Analyzes how mice turn their heads in response to cues at different presentation durations.
    
    Data Structure Overview:
    data_sets = {
        'cue_time1': {
            'total_trials': [all_trials_regardless_of_mouse],
            'trials': {
                'mouse1': {
                    'trials': [all_trials],
                    'successful_trials': [trials_where_mouse_got_it_right],
                    'unsuccessful_trials': [trials_where_mouse_got_it_wrong],
                    'timeouts': [trials_where_mouse_didnt_respond]
                },
                'mouse2': {...}
            }
        },
        'cue_time2': {...}
    }
    """
    
    # Create color map for different cue durations using viridis colormap
    viridis = plt.cm.get_cmap('viridis')
    predefined_colors = {
        'unlimited': viridis(0.0),  # Darkest blue
        '1000ms': viridis(0.1),    
        '750ms': viridis(0.2),      
        '500ms': viridis(0.4),
        '300ms': viridis(0.5),      
        '100ms': viridis(0.6),
        '50ms': viridis(0.7),
        '25ms': viridis(0.8),
        '5ms': viridis(1.0)         # Yellow
    }

    def get_trials(session_list):
        """
        Organizes trials by mouse and filters based on cue_mode.
        Returns both a flat list of all trials and a dict organized by mouse.
        """
        trials = {}     # Structure: {mouse_id: {'trials': [trial_list]}}
        total_trials = []   # Flat list of all trials for statistical purposes
        
        for session in session_list:
            mouse = session.session_dict['mouse_id']

            if mouse in exclusion_mice:
                continue
            
            # Initialize mouse entry if not exists
            if mouse not in trials:
                trials[mouse] = {'trials': []}
            
            # Add trials based on cue_mode filter
            for trial in session.trials:
                should_add = False
                if cue_mode == 'both':
                    should_add = True
                elif cue_mode == 'visual' and 'audio' not in trial['correct_port']:
                    should_add = True
                elif cue_mode == 'audio' and 'audio' in trial['correct_port']:
                    should_add = True
                
                if should_add:
                    trial['session_object'] = session  # Store session reference for later use
                    trials[mouse]['trials'].append(trial)
                    total_trials.append(trial)
                    
        return total_trials, trials

    # Initialize main data structure to hold all trial data organized by cue timing
    data_sets = {}
    
    # Process each session list with its corresponding cue time
    for session_list, cue_time in zip(sessions, cue_times):
        total_trials, trials = get_trials(session_list)
        
        # Initialize cue_time entry if not exists
        if cue_time not in data_sets:
            data_sets[cue_time] = {
                'total_trials': [],
                'trials': defaultdict(list)  # Using defaultdict to auto-initialize mouse lists
            }
        
        # Store trials both in flat list and organized by mouse
        data_sets[cue_time]['total_trials'].extend(total_trials)
        for mouse, trial_list in trials.items():
            data_sets[cue_time]['trials'][mouse].extend(trial_list['trials'])

    # Restructure data for easier access
    for cue_time, data in data_sets.items():
        data_sets[cue_time]['trials'] = {
            mouse: {'trials': trial_list} 
            for mouse, trial_list in data['trials'].items()
        }

    # Categorize trials as successful, unsuccessful, or timeouts for each mouse
    for cue_time in data_sets:
        trials = data_sets[cue_time]['trials']
        for mouse in trials:
            successful_trials = []
            unsuccessful_trials = []
            timeouts = []
            
            for trial in trials[mouse]['trials']:
                if trial["next_sensor"] != {}:  # Mouse made a choice
                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                        successful_trials.append(trial)
                    else:
                        unsuccessful_trials.append(trial)
                else:  # Mouse didn't make a choice
                    timeouts.append(trial)

            # Store categorized trials
            trials[mouse].update({
                'successful_trials': successful_trials,
                'unsuccessful_trials': unsuccessful_trials,
                'timeouts': timeouts
            })

    def get_data(trials, mouse_id):
        """
        Calculate turn percentages for a set of trials.
        Returns the average percentage of turn completed at cue offset.
        """
        percentages = []
        for trial in trials:
            session = trial['session_object']
            
            # Skip trials without video data
            if len(trial['video_frames']) == 0:
                continue
                
            dlc_data = trial.get('DLC_data')
            timestamps = dlc_data['timestamps']
            cue_offset_time = trial['cue_end'] + offset
            start_angle = trial['turn_data']['cue_presentation_angle']
            
            # Find the frame index closest to cue offset time
            index = np.searchsorted(timestamps, cue_offset_time, side='left') 

            # If we found an exact match, use that frame
            # If not, use the last frame before the cue offset
            if index < len(timestamps) and timestamps.values[index] == cue_offset_time:
                # Exact timestamp found, use this index
                pass
            else:
                # No exact match, use the previous frame
                index = index - 1

            # Get coordinates at cue offset
            cue_offset_coords = dlc_data.iloc[index]
            
            # Calculate mouse heading from ear positions
            left_ear = (cue_offset_coords["left_ear"]["x"], cue_offset_coords["left_ear"]["y"])
            left_ear_likeilhood = cue_offset_coords["left_ear"]["likelihood"]
            right_ear = (cue_offset_coords["right_ear"]["x"], cue_offset_coords["right_ear"]["y"])
            right_ear_likelihood = cue_offset_coords["right_ear"]["likelihood"]
            if left_ear_likeilhood < 0.6 or right_ear_likelihood < 0.6:
                continue
            
            # Calculate heading angle
            vector_x = right_ear[0] - left_ear[0]
            vector_y = right_ear[1] - left_ear[1]
            theta_deg = (math.degrees(math.atan2(-vector_y, vector_x)) + 90) % 360
            
            # Calculate new angles relative to ports
            midpoint = ((left_ear[0] + right_ear[0])/2, (left_ear[1] + right_ear[1])/2)
            theta_rad = math.radians(theta_deg)
            
            # Add offset for nose position
            eyes_offset = 40
            new_midpoint = (
                midpoint[0] + eyes_offset * math.cos(theta_rad),
                midpoint[1] - eyes_offset * math.sin(theta_rad)
            )

            # Calculate angles to each port from new position
            port_coordinates = session.port_coordinates
            cue_offset_relative_angles = []
            mouse_heading_rad = np.deg2rad(theta_deg)

            for port_x, port_y in port_coordinates:
                vector_x = port_x - new_midpoint[0]
                vector_y = port_y - new_midpoint[1]
                port_angle_rad = math.atan2(-vector_y, vector_x)
                relative_angle_deg = math.degrees(port_angle_rad - mouse_heading_rad) % 360
                cue_offset_relative_angles.append(relative_angle_deg)

            # Get angle to correct port
            port = int(trial["correct_port"][-1] if trial["correct_port"] != "audio-1" else 1) - 1
            cue_angle = cue_offset_relative_angles[port] % 360
            
            # Normalize angle to [-180, 180]
            if cue_angle > 180:
                cue_angle -= 360
                
            # Calculate turn percentage
            delta_angle = start_angle - cue_angle
            percentage_turn = delta_angle / start_angle
            percentages.append(percentage_turn)

        return np.mean(percentages) * 100 if percentages else 0

    # Initialize data structures for successful and unsuccessful trials
    # These will store the processed turn percentages for each trial type
    # Structure will be:
    # success_data = {
    #     'cue_time1': {
    #         'data': {
    #             'mouse1': [turn_percentage],  # List because we might have multiple values per mouse
    #             'mouse2': [turn_percentage],
    #         },
    #         'mean': average_across_all_mice,
    #         'sem': standard_error_of_mean,
    #         'sd': standard_deviation,
    #         'array': numpy_array_of_all_values
    #     },
    #     'cue_time2': {...}
    # }
    success_data = {}
    unsuccessful_data = {}  # Same structure as success_data

    # Process all trials and calculate statistics for each cue time
    for cue_time in data_sets:
        # Initialize the structure for this cue time
        success_data[cue_time] = {
            'data': {},      # Will hold per-mouse data
            'sem': {},       # Will store standard error of mean
            'sd': {}        # Will store standard deviation
        }
        unsuccessful_data[cue_time] = {
            'data': {},
            'sem': {},
            'sd': {}
        }
        
        trials = data_sets[cue_time]['trials']
        for mouse in trials:
            # Process both successful and unsuccessful trials for this mouse
            # We use a loop to avoid duplicating code since the process is the same
            for data_dict, trial_type in [
                (success_data, 'successful_trials'),
                (unsuccessful_data, 'unsuccessful_trials')
            ]:
                # Calculate turn percentage for this set of trials
                result = get_data(trials[mouse][trial_type], mouse)
                
                # Initialize the mouse's data list if needed
                if mouse not in data_dict[cue_time]['data']:
                    data_dict[cue_time]['data'][mouse] = []
                
                # Store the turn percentage for this mouse
                # We keep it as a list in case we need to store multiple values
                data_dict[cue_time]['data'][mouse].append(result)

    # Calculate summary statistics across all mice for each cue time
    for data_dict in [success_data, unsuccessful_data]:
        for cue_time in data_dict:
            # Extract all turn percentages for this cue time
            # We take val[0] because each mouse's data is stored in a list
            # Even though we usually only have one value per mouse
            data = [val[0] for val in data_dict[cue_time]['data'].values()]
            
            # Update the dictionary with summary statistics
            data_dict[cue_time].update({
                'array': np.array(data),          # All turn percentages as numpy array
                'mean': np.mean(data),            # Average turn percentage across mice
                'sem': np.std(data) / np.sqrt(len(data)),  # Standard error of mean
                'sd': np.std(data)                # Standard deviation
            })
            # Now our dictionary contains both individual mouse data
            # and summary statistics for the group

    # Prepare plotting data
    bin_titles = [ct for ct in predefined_colors.keys() if ct in success_data]
    averages = [success_data[title]['mean'] for title in bin_titles]
    sems = [success_data[title]['sem'] for title in bin_titles]

    # Detect and report outliers
    # outliers = detect_outliers(success_data)
    # if outliers:
    #     print("\nPotential outlier mice detected:")
    #     for cue_time, info in outliers.items():
    #         print(f"\nCue time: {cue_time}")
    #         print(f"Normal range: {info['bounds'][0]:.1f}% to {info['bounds'][1]:.1f}%")
    #         print("Outlier mice:")
    #         for mouse, value in info['outliers'].items():
    #             print(f"  Mouse {mouse}: {value:.1f}%")

    # Create plot with conditional width based on whether legend will be shown
    if plot_individual_mice:
        fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure when showing legend
    else:
        fig, ax = plt.subplots(figsize=(10, 6))  # Normal width when no legend
    x = np.arange(len(bin_titles))

    # Plot individual mouse data if requested
    if plot_individual_mice:
        mouse_data_dict = {}
        for cue_time in success_data:
            for mouse in success_data[cue_time]['data']:
                if mouse not in mouse_data_dict:
                    mouse_data_dict[mouse] = [np.nan] * len(bin_titles)
                for idx, title in enumerate(bin_titles):
                    if mouse in success_data[title]['data']:
                        mouse_data_dict[mouse][idx] = success_data[title]['data'][mouse][0]
        
        # Plot each mouse's data
        for mouse, mouse_data in mouse_data_dict.items():
            ax.plot(x, mouse_data, label=f'Mouse {mouse}', linestyle='--', marker='o')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside

    # Plot average line and error shading
    ax.plot(x, averages, color=(0, 0.68, 0.94), marker='o', linestyle='-', lw=2)
    ax.fill_between(x, 
                    np.array(averages) - np.array(sems),
                    np.array(averages) + np.array(sems),
                    color=lighten_color((0, 0.68, 0.94)), 
                    alpha=0.2)

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels(bin_titles, rotation=0, ha="center", fontsize=12)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(plot_title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save figures if output_path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if draft:
            base_filename = f"{date_time}_{plot_save_name}_{cue_mode}"
        else:
            base_filename = f"final_{plot_save_name}_{cue_mode}"
        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"

        # Check for existing files and modify filenames if necessary
        counter = 0
        while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
            output_filename_svg = f"{base_filename}_{counter}.svg"
            output_filename_png = f"{base_filename}_{counter}.png"
            counter += 1

        # Save the plot as SVG in the desired folder
        print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
        plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

        # Save the plot as PNG in the desired folder
        print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
        plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)

    plt.show()

def detect_outliers(data_dict):
    """
    Detects outlier mice for each cue time using the IQR method.
    Returns a dictionary of outlier mice and their values for each cue time.
    """
    outliers = {}
    
    for cue_time in data_dict:
        # Get data for this cue time
        mouse_data = {
            mouse: values[0] 
            for mouse, values in data_dict[cue_time]['data'].items()
        }
        
        if len(mouse_data) < 4:  # Need at least 4 points for meaningful outlier detection
            continue
            
        values = np.array(list(mouse_data.values()))
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        cue_outliers = {
            mouse: value 
            for mouse, value in mouse_data.items()
            if value < lower_bound or value > upper_bound
        }
        
        if cue_outliers:
            outliers[cue_time] = {
                'outliers': cue_outliers,
                'bounds': (lower_bound, upper_bound),
                'quartiles': (q1, q3)
            }
    
    return outliers