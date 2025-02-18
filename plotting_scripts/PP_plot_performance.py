from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

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
                              output_path = None,
                              plot_individual_mice = False,
                              exclusion_mice = []):
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
        num_bins = 6
    else:
        limits = (-180, 180)

    angle_range = limits[1] - limits[0]

    bin_size = round(angle_range / num_bins)
    bin_titles = []
    performance = []


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
                if trial.get("turn_data") != None:
                    angle = trial["turn_data"]["cue_presentation_angle"]
                    if trial["turn_data"]["left_ear_likelihood"] < 0.6:
                        continue
                    if trial["turn_data"]["right_ear_likelihood"] < 0.6:
                        continue
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

    # Radial plot:

    if plot_mode == 'radial':

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

        # Dictionary to hold mouse-specific data
        if plot_individual_mice:
            mouse_data_dict = {cue_group: {} for cue_group in cue_modes}

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

            # Store individual mouse data if needed
            if plot_individual_mice:
                for mouse in data_sets['mice']:
                    if cue_group in data_sets['mice'][mouse]:
                        mouse_data = data_sets['mice'][mouse][cue_group]['performance']
                        if mouse not in mouse_data_dict[cue_group]:
                            mouse_data_dict[cue_group][mouse] = []
                        mouse_data_dict[cue_group][mouse].append(mouse_data)

        # Plot individual mouse data if requested
        if plot_individual_mice:
            for cue_group in cue_modes:
                for mouse, mouse_data_list in mouse_data_dict[cue_group].items():
                    for mouse_data in mouse_data_list:
                        mouse_data = np.append(mouse_data, mouse_data[0])  # Close the circular plot for individual mouse data
                        ax.plot(angles_rad, mouse_data, label=f"Mouse {mouse}", linestyle='--', marker='o')

            ax.legend(loc='upper right')

        # Set the tick locations and labels
        tick_locs = np.radians(np.arange(limits[0], limits[1] + 1, 30)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1] + 1, 30)]
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
        
        # Change legend position by coords
        ax.legend(loc=(0.85, 0.9))

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

        # Show the plot
        plt.show()


    if plot_mode == 'linear_comparison':

        # Prepare the angles as a numeric sequence
        angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180

        # Create a line plot
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

        # Dictionary to hold mouse-specific data
        if plot_individual_mice:
            mouse_data_dict = {cue_group: {} for cue_group in cue_modes}

        # Iterate through the plotting data dictionary to plot both normal and catch data
        for cue_group in cue_modes:
            performance = plotting_data[cue_group]['performance']
            performance_sd = plotting_data[cue_group]['performance_sd']
            performance_sem = plotting_data[cue_group]['performance_sem']
            bin_titles = plotting_data[cue_group]['bin_titles']

            # Prepare the angles as a numeric sequence (linear plot, so no need for radians)
            angles_deg = np.array(bin_titles, dtype=np.float64)
            performance_data = np.array(performance)

            # Plot the performance data line
            ax.plot(angles_deg, performance_data, marker='o', color=colors[cue_group], label=cue_group.capitalize())

            # Add the shaded area for standard deviation or SEM
            if error_bars == 'SD':
                ax.fill_between(angles_deg, performance_data - performance_sd, performance_data + performance_sd, 
                                color=lighten_color(colors[cue_group]), alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            elif error_bars == 'SEM':
                ax.fill_between(angles_deg, performance_data - performance_sem, performance_data + performance_sem, 
                                color=lighten_color(colors[cue_group]), alpha=0.4)
                ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

            # Store individual mouse data if needed
            if plot_individual_mice:
                for mouse in data_sets['mice']:
                    if cue_group in data_sets['mice'][mouse]:
                        mouse_data = data_sets['mice'][mouse][cue_group]['performance']
                        if mouse not in mouse_data_dict[cue_group]:
                            mouse_data_dict[cue_group][mouse] = []
                        mouse_data_dict[cue_group][mouse].append(mouse_data)

        # Plot individual mouse data if requested
        if plot_individual_mice:
            for cue_group in cue_modes:
                for mouse, mouse_data_list in mouse_data_dict[cue_group].items():
                    for mouse_data in mouse_data_list:
                        ax.plot(angles_deg, mouse_data, label=f"Mouse {mouse}", linestyle='--', marker='o')

            ax.legend(loc='upper right')

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

        # Show the plot
        plt.tight_layout()
        plt.show()


