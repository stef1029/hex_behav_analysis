from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np
from datetime import datetime

# Define your colors
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

# Function to lighten a color for shaded regions
def lighten_color(color, factor=0.5):
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def plot_performance_by_angle(sessions, 
                              title = 'title', 
                              bin_mode = 'manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_modes=['all_trials'],
                              error_bars = 'SEM',
                              output_path = None):
    """
    This function takes a list of sessions and plots the performance by angle of all trials in the sessions given.
    """
    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]

    def get_trials(sessions):
        # Load trial list:
        mice = {}
        total_trials = {'all_trials': [],
                        'visual_trials': [],
                        'audio_trials': []}
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')  
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

    data_sets = {}
    # data_sets will look like this:
    # data_sets = {
    #     'total_trials': {'all_trials': [trial1, trial2, trial3, ...],
    #                     'visual_trials': [trial1, trial2, trial3, ...],
    #                     'audio_trials': [trial1, trial2, trial3, ...]},
    #     'mice': {
    #         'mouse1': {
#                 'all_trials': {
#                     'trials': [trial1, trial2, trial3, ...],
#                     'performance': [performance1, performance2, performance3, ...],
#                     'n': [n1, n2, n3, ...]
#                 'visual_trials': {
#                     'trials': [trial1, trial2, trial3, ...],
#                     'performance': [performance1, performance2, performance3, ...],
#                     'n': [n1, n2, n3, ...]
#                 'audio_trials': {
#                     'trials': [trial1, trial2, trial3, ...],
#                     'performance': [performance1, performance2, performance3, ...],
#                     'n': [n1, n2, n3, ...
    #             },
    #         'mouse2: etc...
    #         },

    data_sets['mice'], data_sets['total_trials'] = get_trials(sessions)

    n = len(data_sets['total_trials'])
    if bin_mode == 'manual':
        num_bins = num_bins
    elif bin_mode == 'rice':
        num_bins = 2 * n ** (1/3)
    elif bin_mode == 'tpb':
        num_bins = n / trials_per_bin
    else:
        raise ValueError('bin_mode must be "manual", "rice" or "tpb"')

    if plot_mode == 'linear_comparison' \
        or plot_mode == 'bar_split' \
            or plot_mode == 'bar_split_overlay':
        limits = (0, 180)
        num_bins = 10
    else:
        limits = (-180, 180)

    angle_range = limits[1] - limits[0]

    bin_size = round(angle_range / num_bins)
    bin_titles = []
    performance = []

    if plot_mode == 'bar_split' or plot_mode == 'bar_split_overlay':
        for mouse in data_sets['mice']:
            left_bins = {i: [] for i in range(limits[0], limits[1], bin_size)}
            right_bins = {i: [] for i in range(limits[0], limits[1], bin_size)}

            for trial in data_sets['mice'][mouse]['trials']:
                if trial["turn_data"] is not None:
                    angle = trial["turn_data"]["cue_presentation_angle"]
                    if trial["next_sensor"] != {}:
                        correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                    else:
                        correct = 0
                    if angle < 0:
                        bin_index = abs(angle) // bin_size * bin_size
                        left_bins[bin_index].append(correct)
                    elif angle > 0:
                        bin_index = angle // bin_size * bin_size
                        right_bins[bin_index].append(correct)

            data_sets['mice'][mouse]['left_performance'] = calc_performance(left_bins)
            data_sets['mice'][mouse]['right_performance'] = calc_performance(right_bins)
            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(left_bins)]

        left_performance_data = np.array([data_sets['mice'][mouse]['left_performance'] for mouse in data_sets['mice']])
        left_performance = np.mean(left_performance_data, axis=0)
        left_performance_sd = np.std(left_performance_data, axis=0)
        n = len(left_performance_data)
        left_performance_sem = left_performance_sd / np.sqrt(n)

        right_performance_data = np.array([data_sets['mice'][mouse]['right_performance'] for mouse in data_sets['mice']])
        right_performance = np.mean(right_performance_data, axis=0)
        right_performance_sd = np.std(right_performance_data, axis=0)
        n = len(left_performance_data)
        right_performance_sem = right_performance_sd / np.sqrt(n)

    else:
        plotting_data = {f'{cue_group}': {'performance': [], 
                                          'performance_sd': [],
                                          'performance_sem': [],
                                          'length': [],
                                          'n': []} 
                                            for cue_group in cue_modes}
        
        for cue_group in cue_modes:
            for mouse in data_sets['mice']:
                bins = {i: [] for i in range(limits[0], limits[1], bin_size)}

                for trial in data_sets['mice'][mouse][cue_group]['trials']:
                    # print(trial)
                    if trial["turn_data"] != None:
                        angle = trial["turn_data"]["cue_presentation_angle"]
                        for bin in bins:
                            if angle < bin + bin_size and angle >= bin:
                                if trial["next_sensor"] != {}:
                                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                        bins[bin].append(1)
                                    else:
                                        bins[bin].append(0)
                                else:
                                    bins[bin].append(0)

                # caluclate the total lenth of each bin across mice:
                data_sets['mice'][mouse][cue_group]['performance'] = calc_performance(bins)
                data_sets['mice'][mouse][cue_group]['n'] = [len(bins[key]) for key in sorted(bins)]
                # print(trials[mouse]['n'])
                bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

            length_data = np.array([data_sets['mice'][mouse][cue_group]['n'] for mouse in data_sets['mice']])
            length = np.mean(length_data, axis=0)
            performance_data = np.array([data_sets['mice'][mouse][cue_group]['performance'] for mouse in data_sets['mice']])
            performance = np.mean(performance_data, axis=0)
            performance_sd = np.std(performance_data, axis=0)
            n = len(performance_data)
            performance_sem = performance_sd / np.sqrt(n)

            plotting_data[cue_group]['performance'] = performance
            plotting_data[cue_group]['performance_sd'] = performance_sd
            plotting_data[cue_group]['performance_sem'] = performance_sem
            plotting_data[cue_group]['length'] = length
            plotting_data[cue_group]['n'] = n
            plotting_data[cue_group]['bin_titles'] = bin_titles

    def plot_performance(bin_titles, performance, errors, title, color_key):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        color = colors[color_key]
        lighter_color = lighten_color(color)

        bin_numeric = np.array(bin_titles, dtype=float)

        plt.errorbar(bin_numeric, performance, yerr=errors, fmt='o-', color=color, ecolor=lighter_color, elinewidth=3, capsize=0, linestyle='-', linewidth=2)
        plt.xlabel('Turn Angle (degrees)', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.title(title, fontsize=16)

        plt.xticks(bin_numeric, bin_titles, rotation=45)
        plt.xlim(limits[0], limits[1])
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_performance_multi(bin_titles, left_performance, left_errors, right_performance, right_errors, left_title, right_title):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        bin_numeric = np.array(bin_titles, dtype=float)

        plt.errorbar(bin_numeric, left_performance, yerr=left_errors, fmt='o-', color=colors['Cyan'], ecolor=lighten_color(colors['Cyan']), elinewidth=3, capsize=0, linestyle='-', linewidth=2, label=left_title)
        plt.errorbar(bin_numeric, right_performance, yerr=right_errors, fmt='o-', color=colors['Orange'], ecolor=lighten_color(colors['Orange']), elinewidth=3, capsize=0, linestyle='-', linewidth=2, label=right_title)

        plt.xlabel('Turn Angle (degrees)', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.title(title, fontsize=16)

        plt.xticks(bin_numeric, bin_titles, rotation=45)
        plt.xlim(limits[0], limits[1])
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if plot_mode == 'bar_split':
        plot_performance(bin_titles, left_performance, left_performance_sem, 'Left Turn Performance', 'Cyan')
        plot_performance(bin_titles, right_performance, right_performance_sem, 'Right Turn Performance', 'Orange')

    if plot_mode == 'bar_split_overlay':
        plot_performance_multi(bin_titles, left_performance, left_performance_sem, right_performance, right_performance_sem, 'Left Turn Performance', 'Right Turn Performance')

    # Radial plot:

    if plot_mode == 'radial':

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

        for cue_group in cue_modes:
            performance = plotting_data[cue_group]['performance']
            performance_sd = plotting_data[cue_group]['performance_sd']
            performance_sem = plotting_data[cue_group]['performance_sem']
            bin_titles = plotting_data[cue_group]['bin_titles']

            # Prepare the angles as a numeric sequence
            angles_deg = np.array(bin_titles, dtype=np.float64)
            performance_data = np.array(performance)

            # Adjust angles for plotting and convert to radians
            adjusted_angles_deg = angles_deg % 360
            angles_rad = np.radians(adjusted_angles_deg)

            # Append the first element to close the circular plot
            angles_rad = np.append(angles_rad, angles_rad[0])
            performance_data = np.append(performance_data, performance_data[0])
            performance_sem = np.append(performance_sem, performance_sem[0])
            

            # Plot the performance data line
            ax.plot(angles_rad, performance_data, marker='o', color=colors[cue_group], label=cue_group.capitalize())

            # Add the shaded area for standard deviation or SEM
            if error_bars == 'SD':
                ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, 
                                color=lighten_color(colors[cue_group]), alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            elif error_bars == 'SEM':
                ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, 
                                color=lighten_color(colors[cue_group]), alpha=0.4)
                ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Set the tick locations and labels
        tick_locs = np.radians(np.arange(limits[0], limits[1]+1, 30)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1]+1, 30)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)

        # Set other plot properties
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)

        # Add text with trial and mice info
        text = f"Trials: {len(data_sets['total_trials']['all_trials'])} - Mice: {len(data_sets['mice'])}"
        ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Add title
        ax.set_title(title, va='bottom', fontsize=16)
        # change legend position by coords:
        ax.legend(loc=(0.85, 0.9))

        sub_dir_name = 'base_performance_plots'
        # Check if the directory exists in the output path, and create it if it doesn't
        final_output_path = output_path / sub_dir_name  # Create the full path
        if not final_output_path.exists():
            final_output_path.mkdir(parents=True)  # Create the directory, including any missing parents

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cue_modes_str = '_'.join(cue_modes)  # Join list elements into a string
        output_filename = f"{date_time}_plot_performance_by_angle_radial_{cue_modes_str}.svg"

        # Check for existing files and modify filename if necessary
        counter = 0
        while (final_output_path / output_filename).exists():
            output_filename = f"{date_time}_plot_performance_by_angle_radial_{cue_modes_str}_{counter}.svg"
            counter += 1

        # Save the plot as SVG in the desired folder
        print(f"Saving plot to: '{final_output_path / output_filename}'")
        plt.savefig(final_output_path / output_filename, format='svg', bbox_inches='tight')

        # Show the plot
        plt.show()


    if plot_mode == 'linear_comparison':

        # Prepare the angles as a numeric sequence
        angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180

        # Create a line plot
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

        # Iterate through the plotting data dictionary to plot both normal and catch data
        for key, data in plotting_data.items():
            if key == 'p_values':
                continue
            performance_data = data['performance_mean']
            performance_sem = data['performance_sem']
            performance_sd = data['performance_sd']

            # Plot the line for this dataset
            ax.plot(angles_deg, performance_data, marker='o', color=colors[key], label=key.capitalize())

            # Add the shaded region for standard deviation or SEM
            if error_bars == 'SD':
                ax.fill_between(angles_deg, performance_data - performance_sd, performance_data + performance_sd, color=lighten_color(colors[key]), alpha=0.4)
            elif error_bars == 'SEM':
                ax.fill_between(angles_deg, performance_data - performance_sem, performance_data + performance_sem, color=lighten_color(colors[key]), alpha=0.4)

        # Add stars for significant p-values
        p_values = plotting_data['p_values']
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                # Find the maximum y value (performance) for this point to place the star above
                max_performance = max(plotting_data['normal']['performance_mean'][i], plotting_data['catch']['performance_mean'][i])

                # Add a star above the data point (adjust the vertical position as needed)
                ax.text(angles_deg[i] + 10, max_performance + 0.05, f'* (p={round(p_val, 3)})', fontsize=14, color='black', ha='center')

        # Labeling the axes
        ax.set_xlabel('Turn Angle (degrees)', fontsize=14)
        ax.set_ylabel('Performance', fontsize=14)

        # Set the x-ticks to correspond to the bin_titles
        ax.set_xticks(angles_deg)  # Set x-ticks to the center of each bin
        ax.set_xticklabels(angles_deg, rotation=45)  # Use bin titles as labels

        # Set limits for x and y axes
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(0, 1)

        # Add a grid and legend
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.legend()

        # Add text at the bottom right
        text = f"Trials: {len(data_sets['total_trials']['all_trials'])} - Mice: {len(data_sets['mice'])}"
        ax.text(0.21, 0.05, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')

        # Add a title
        ax.set_title(title, fontsize=16)


    
        sub_dir_name = 'base_performance_plots'
        # Check if the directory exists in the output path, and create it if it doesn't
        final_output_path = output_path / sub_dir_name  # Create the full path
        if not final_output_path.exists():
            final_output_path.mkdir(parents=True)  # Create the directory, including any missing parents

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cue_modes_str = '_'.join(cue_modes)  # Join list elements into a string
        output_filename = f"{date_time}_plot_performance_by_angle_linear_{cue_modes_str}.svg"

        # Check for existing files and modify filename if necessary
        counter = 0
        while (final_output_path / output_filename).exists():
            output_filename = f"{date_time}_plot_performance_by_angle_linear_{cue_modes_str}_{counter}.svg"
            counter += 1

        # Save the plot as SVG in the desired folder
        print(f"Saving plot to: '{final_output_path / output_filename}'")
        plt.savefig(final_output_path / output_filename, format='svg', bbox_inches='tight')

        # Show the plot
        plt.tight_layout()
        plt.show()


