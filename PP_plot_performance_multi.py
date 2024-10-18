from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np
from datetime import datetime
import matplotlib.cm as cm

# Define your colors
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

# Function to lighten a color for shaded regions
def lighten_color(color, factor=0.5):
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def plot_performance_by_angle(sessions_input, 
                              title='title', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_modes=['all_trials'],
                              error_bars='SEM',
                              output_path=None,
                              plot_individual_mice=False,
                              exclusion_mice=[]):
    """
    This function takes a list of sessions or a dictionary of session lists and plots the performance by angle.
    If a dictionary is provided, cue_modes must be ['all_trials'], and each key-value pair in the dictionary
    is plotted as an individual line with its corresponding label.
    """
    # Function to calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]

    # Function to retrieve trials from sessions
    def get_trials(sessions):
        # Load trial list:
        mice = {}
        total_trials = {'all_trials': [],
                        'visual_trials': [],
                        'audio_trials': []}
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')  
            if mouse in exclusion_mice:
                continue
            if mouse not in mice:
                mice[mouse] = {'all_trials': {'trials': []},
                               'visual_trials': {'trials': []},
                               'audio_trials': {'trials': []}}
            for trial in session.trials:
                if not trial.get('catch', False):
                    mice[mouse]['all_trials']['trials'].append(trial)
                    total_trials['all_trials'].append(trial)

                    if 'audio' not in trial.get('correct_port', ''):
                        mice[mouse]['visual_trials']['trials'].append(trial)
                        total_trials['visual_trials'].append(trial)

                    if 'audio' in trial.get('correct_port', ''):
                        mice[mouse]['audio_trials']['trials'].append(trial)
                        total_trials['audio_trials'].append(trial)
        
        return mice, total_trials

    # Start plot
    if plot_mode == 'radial':
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    else:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

    # Determine whether sessions_input is a dict or a list
    if isinstance(sessions_input, dict):
        # Enforce that cue_modes must be ['all_trials']
        if cue_modes != ['all_trials']:
            raise ValueError("When providing a sessions dictionary, cue_modes must be ['all_trials'].")
        
        # Proceed with dictionary logic
        sessions_dict = sessions_input

        # Determine number of sessions and colors
        number_of_sessions = len(sessions_dict)
        session_titles = list(sessions_dict.keys())
        session_lists = list(sessions_dict.values())

        def get_colors(number_of_sessions):
            if number_of_sessions <= 3:
                # Use the predefined colors
                color_values = [colors['all_trials'], colors['visual_trials'], colors['audio_trials']]
                return color_values[:number_of_sessions]
            else:
                # Use the viridis color palette
                cmap = plt.cm.get_cmap('viridis', number_of_sessions)
                color_values = [cmap(i) for i in range(number_of_sessions)]
                return color_values

        colors_list = get_colors(number_of_sessions)

        # Loop through each session list
        for idx, (session_title, sessions) in enumerate(sessions_dict.items()):
            data_sets = {}
            data_sets['mice'], data_sets['total_trials'] = get_trials(sessions)

            n = len(data_sets['total_trials']['all_trials'])
            if bin_mode == 'manual':
                num_bins_used = num_bins
            elif bin_mode == 'rice':
                num_bins_used = int(2 * n ** (1/3))
            elif bin_mode == 'tpb':
                num_bins_used = int(n / trials_per_bin)
            else:
                raise ValueError('bin_mode must be "manual", "rice" or "tpb"')

            if plot_mode in ['linear_comparison', 'bar_split', 'bar_split_overlay']:
                limits = (0, 180)
                num_bins_used = 6
            else:
                limits = (-180, 180)

            angle_range = limits[1] - limits[0]
            bin_size = angle_range / num_bins_used
            bin_titles = []
            performance = []

            plotting_data = {cue_group: {'performance': [], 
                                         'performance_sd': [],
                                         'performance_sem': [],
                                         'length': [],
                                         'n': []} 
                             for cue_group in cue_modes}
            
            for cue_group in cue_modes:
                for mouse in data_sets['mice']:
                    bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                    for trial in data_sets['mice'][mouse][cue_group]['trials']:
                        if trial["turn_data"] is not None:
                            angle = trial["turn_data"]["cue_presentation_angle"]
                            for bin in bins:
                                if bin <= angle < bin + bin_size:
                                    if trial["next_sensor"]:
                                        if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                            bins[bin].append(1)
                                        else:
                                            bins[bin].append(0)
                                    else:
                                        bins[bin].append(0)

                    data_sets['mice'][mouse][cue_group]['performance'] = calc_performance(bins)
                    data_sets['mice'][mouse][cue_group]['n'] = [len(bins[key]) for key in sorted(bins)]
                    bin_titles = [f"{bin + (bin_size / 2)}" for bin in sorted(bins)]

                length_data = np.array([data_sets['mice'][mouse][cue_group]['n'] for mouse in data_sets['mice']])
                length = np.mean(length_data, axis=0)
                performance_data = np.array([data_sets['mice'][mouse][cue_group]['performance'] for mouse in data_sets['mice']])
                performance = np.mean(performance_data, axis=0)
                performance_sd = np.std(performance_data, axis=0)
                n_mice = len(performance_data)
                performance_sem = performance_sd / np.sqrt(n_mice)

                plotting_data[cue_group]['performance'] = performance
                plotting_data[cue_group]['performance_sd'] = performance_sd
                plotting_data[cue_group]['performance_sem'] = performance_sem
                plotting_data[cue_group]['length'] = length
                plotting_data[cue_group]['n'] = n_mice
                plotting_data[cue_group]['bin_titles'] = bin_titles

                if plot_mode == 'radial':
                    # Radial plot for each session list
                    performance = plotting_data[cue_group]['performance']
                    performance_sd = plotting_data[cue_group]['performance_sd']
                    performance_sem = plotting_data[cue_group]['performance_sem']
                    bin_titles = plotting_data[cue_group]['bin_titles']

                    angles_deg = np.array(bin_titles, dtype=np.float64)
                    performance_data = np.array(performance)

                    adjusted_angles_deg = angles_deg % 360
                    angles_rad = np.radians(adjusted_angles_deg)

                    angles_rad = np.append(angles_rad, angles_rad[0])
                    performance_data = np.append(performance_data, performance_data[0])
                    performance_sem = np.append(performance_sem, performance_sem[0])

                    color = colors_list[idx % len(colors_list)]
                    ax.plot(angles_rad, performance_data, marker='o', label=session_title, color=color)

                    if error_bars == 'SEM':
                        ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem,
                                        alpha=0.3, color=lighten_color(color))

                elif plot_mode == 'linear_comparison':
                    # Linear comparison plot for each session list
                    angles_deg = np.array(bin_titles, dtype=np.float64)
                    performance_data = np.array(performance)

                    color = colors_list[idx % len(colors_list)]
                    ax.plot(angles_deg, performance_data, marker='o', label=session_title, color=color)

                    if error_bars == 'SEM':
                        ax.fill_between(angles_deg, performance_data - performance_sem, performance_data + performance_sem,
                                        alpha=0.3, color=lighten_color(color))



    # Final plot adjustments and show
    ax.legend(loc='upper right')
    ax.set_title(title)
    
    if plot_mode == 'radial':
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_xlim(np.radians(limits[0]), np.radians(limits[1]))
        ax.set_ylim(0, 1)

    elif plot_mode == 'linear_comparison':
        ax.set_xlabel('Turn Angle (degrees)')
        ax.set_ylabel('Performance')
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(0, 1)

    # ------ save figures ------    

    if output_path is not None:
        # Create directory if it doesn't exist (but don't concatenate path multiple times)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        cue_modes_str = '_'.join(cue_modes)  # Join list elements into a string
        base_filename = f"{date_time}_angular_performance_line_{cue_modes_str}"
        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"

        # Check for existing SVG and PNG files and modify filenames if necessary
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

    # --------------------------------------------

    plt.show()
