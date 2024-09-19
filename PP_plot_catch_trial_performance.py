from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np
from scipy.stats import ttest_rel
from datetime import datetime

# Define your colors
colors = {
    "normal": (0, 0.68, 0.94),
    "catch": (0.93, 0, 0.55)
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
                              cue_mode='both',
                              error_bars = 'SEM',
                              output_path = None):
    """
    This function takes a list of sessions and plots the performance by angle of all trials in the sessions given.
    ### Inputs: 
    - cohort: Cohort_folder object
    - sessions: list of session objects (Session class already loaded)
    - title: title of the plot
    - bin_mode: 'manual' (->set num_bins), 'rice', 'tpb' (trials per bin ->set tpb value) -  to choose the method of binning
    - num_bins: number of bins to divide the angles into
    - trials_per_bin: number of trials per bin for tbp bin mode.
    - plot_mode: 'radial' or 'bar' to choose the type of plot
    - cue_mode: 'both', 'visual' or 'audio' to choose the type of cue to plot
    """
    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
        # returns a list of averages for each key in bins
    
    # Load trial list:
    total_trials = []
    for session in sessions:
        if cue_mode == 'both':
            total_trials += session.trials
        elif cue_mode == 'visual':
            for trial in session.trials:
                if 'audio' not in trial['correct_port']:
                    total_trials.append(trial)
        elif cue_mode == 'audio':
            for trial in session.trials:
                if 'audio' in trial['correct_port']:
                    total_trials.append(trial)

    trials = {}
    for session in sessions:
        mouse = session.session_dict['mouse_id']
        if mouse not in trials:
            trials[mouse] = {'trials': []}
        if cue_mode == 'both':
            trials[mouse]['trials'] += session.trials
        elif cue_mode == 'visual':
            for trial in session.trials:
                if 'audio' not in trial['correct_port']:
                    trials[mouse]['trials'].append(trial)
        elif cue_mode == 'audio':
            for trial in session.trials:
                if 'audio' in trial['correct_port']:
                    trials[mouse]['trials'].append(trial)


    if plot_mode == 'linear_comparison' \
        or plot_mode == 'bar_split' \
            or plot_mode == 'bar_split_overlay':
        limits = (0, 180)
        num_bins = 10
    else:
        limits = (-180, 180)

    angle_range = limits[1] - limits[0]

    # bin the trials into 30 degree bins, ranging from 180 to -180
    n = len(total_trials)
    if bin_mode == 'manual':
        num_bins = num_bins
    elif bin_mode == 'rice':
        num_bins = 2 * n ** (1/3)
    elif bin_mode == 'tpb':
        num_bins = n / trials_per_bin
    else:
        raise ValueError('bin_mode must be "manual", "rice" or "tpb"')

    bin_size = round(angle_range / num_bins)

    bin_titles = []
    performance = []
    
    if plot_mode == 'bar_split' or plot_mode == 'bar_split_overlay':

        for mouse in trials:

            left_bins = {i: [] for i in range(limits[0], limits[1], bin_size)}
            right_bins = {i: [] for i in range(limits[0], limits[1], bin_size)}

            # Bin trials based on turn direction and angle
            for trial in trials[mouse]['trials']:
                if trial["turn_data"] is not None:

                    angle = trial["turn_data"]["cue_presentation_angle"]
                    if trial["next_sensor"] != {}:
                        correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                    else:
                        correct = 0
                        
                    if angle < 0:  # Left turn
                        bin_index = abs(angle) // bin_size * bin_size
                        left_bins[bin_index].append(correct)
                    elif angle > 0:  # Right turn
                        bin_index = angle // bin_size * bin_size
                        right_bins[bin_index].append(correct)

            trials[mouse]['left_performance'] = calc_performance(left_bins)
            trials[mouse]['right_performance'] = calc_performance(right_bins)
            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(left_bins)] 

        left_performance_data = np.array([trials[mouse]['left_performance'] for mouse in trials])
        left_performance = np.mean(left_performance_data, axis=0)
        left_performance_sd = np.std(left_performance_data, axis=0)
        n = len(left_performance_data)
        left_performance_sem = left_performance_sd / np.sqrt(n)

        right_performance_data = np.array([trials[mouse]['right_performance'] for mouse in trials])
        right_performance = np.mean(right_performance_data, axis=0)
        right_performance_sd = np.std(right_performance_data, axis=0)
        n = len(left_performance_data)
        right_performance_sem = right_performance_sd / np.sqrt(n)

    else:
        for mouse in trials:
            bins_normal = {i: [] for i in range(limits[0], limits[1], bin_size)}
            bins_catch = {i: [] for i in range(limits[0], limits[1], bin_size)}

            catch_count = 0
            normal_count = 0

            for trial in trials[mouse]['trials']:
                if trial["turn_data"] != None:
                    if plot_mode == 'linear_comparison':
                        angle = abs(trial["turn_data"]["cue_presentation_angle"])
                    else:
                        angle = trial["turn_data"]["cue_presentation_angle"]
                    for bin in bins_normal:
                        if angle < bin + bin_size and angle >= bin:
                            if trial['catch'] == True:
                                catch_count += 1
                                if trial["next_sensor"] != {}:
                                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                        bins_catch[bin].append(1)
                                    else:
                                        bins_catch[bin].append(0)
                                else:
                                    bins_catch[bin].append(0)
                            else:
                                normal_count +=1
                                if trial["next_sensor"] != {}:
                                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                        bins_normal[bin].append(1)
                                    else:
                                        bins_normal[bin].append(0)
                                else:
                                    bins_normal[bin].append(0)
            # print(catch_count, normal_count)

            trials[mouse]['normal_performance'] = calc_performance(bins_normal)
            trials[mouse]['catch_performance'] = calc_performance(bins_catch)
            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins_normal)]

        # Initialize the plotting_data dictionary
        plotting_data = {}

        # Process and store normal performance data
        data_normal = {}
        normal_performance_data = np.array([trials[mouse]['normal_performance'] for mouse in trials])
        data_normal['performance_data'] = normal_performance_data
        data_normal['performance_mean'] = np.mean(normal_performance_data, axis=0)
        data_normal['performance_sd'] = np.std(normal_performance_data, axis=0)
        data_normal['n'] = len(normal_performance_data)     # n mice
        data_normal['performance_sem'] = data_normal['performance_sd'] / np.sqrt(data_normal['n'])

        # Store normal data in the plotting_data dictionary
        plotting_data['normal'] = data_normal

        # Process and store catch performance data
        data_catch = {}
        catch_performance_data = np.array([trials[mouse]['catch_performance'] for mouse in trials])
        data_catch['performance_data'] = catch_performance_data
        data_catch['performance_mean'] = np.mean(catch_performance_data, axis=0)
        data_catch['performance_sd'] = np.std(catch_performance_data, axis=0)
        data_catch['n'] = len(catch_performance_data)       # n mice
        data_catch['performance_sem'] = data_catch['performance_sd'] / np.sqrt(data_catch['n'])

        # Store catch data in the plotting_data dictionary
        plotting_data['catch'] = data_catch

        # Ensure the number of data points (mice) in both normal and catch are the same
        assert len(plotting_data['normal']['performance_data']) == len(plotting_data['catch']['performance_data']), "Mismatch in number of mice"

        # Perform paired t-tests for each bin and store p-values
        p_values = []
        for bin_index in range(plotting_data['normal']['performance_data'].shape[1]):  # Loop through each bin
            normal_bin_data = plotting_data['normal']['performance_data'][:, bin_index]
            catch_bin_data = plotting_data['catch']['performance_data'][:, bin_index]

            # Perform a paired t-test between normal and catch data for this bin
            t_stat, p_val = ttest_rel(normal_bin_data, catch_bin_data)

            # Store the p-value
            p_values.append(p_val)

        # Convert the p-values list to a numpy array
        p_values = np.array(p_values)

        # Add the p-values to the plotting_data dictionary
        plotting_data['p_values'] = p_values



    def plot_performance(bin_titles, performance, errors, title, color_map='viridis'):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        # Use the color map for the line color
        colors = plt.cm.get_cmap(color_map, len(bin_titles))

        # Convert bin titles to numeric if they are not already, for plotting
        bin_numeric = np.array(bin_titles, dtype=float)

        # Create a line plot with error bars
        plt.errorbar(bin_numeric, performance, yerr=errors, fmt='o-', color='royalblue', ecolor='lightsteelblue', elinewidth=3, capsize=0, linestyle='-', linewidth=2)
        plt.text(0, 0, '±SEM', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='black')
        plt.xlabel('Turn Angle (degrees)', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.title(title, fontsize=16)

        # Set the x-ticks to correspond to bin_titles
        plt.xticks(bin_numeric, bin_titles, rotation=45)
        plt.xlim(0, 180)
        plt.ylim(0, 1)

        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_performance_multi(bin_titles, left_performance, left_errors, right_performance, right_errors, left_title, right_title, color_map='viridis'):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        # Use the color map for the line color
        colors = plt.cm.get_cmap(color_map, len(bin_titles))

        # Convert bin titles to numeric if they are not already, for plotting
        bin_numeric = np.array(bin_titles, dtype=float)

        # Create a line plot with error bars
        plt.errorbar(bin_numeric, left_performance, yerr=left_errors, fmt='o-', color='royalblue', ecolor='lightsteelblue', elinewidth=3, capsize=0, linestyle='-', linewidth=2, label=left_title)
        plt.errorbar(bin_numeric, right_performance, yerr=right_errors, fmt='o-', color='darkorange', ecolor='moccasin', elinewidth=3, capsize=0, linestyle='-', linewidth=2, label=right_title)
        plt.text(0, -0.1, '±SEM', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='black')
        plt.xlabel('Turn Angle (degrees)', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.title(title, fontsize=16)

        # Set the x-ticks to correspond to bin_titles
        plt.xticks(bin_numeric, bin_titles, rotation=45)
        plt.xlim(limits[0], limits[1])
        plt.ylim(0, 1)

        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if plot_mode == 'bar_split':
        
        plot_performance(bin_titles, left_performance, left_performance_sem, 'Left Turn Performance')
        plot_performance(bin_titles, right_performance, right_performance_sem, 'Right Turn Performance')

    if plot_mode == 'bar_split_overlay':
        plot_performance_multi(bin_titles, left_performance, left_performance_sem, right_performance, right_performance_sem, 'Left Turn Performance', 'Right Turn Performance')

    # Bar plot:
    if plot_mode == 'bar':

        plot_performance(bin_titles, performance, performance_sem, title)

    # Radial Plot:

    if plot_mode == 'radial':

        # Preparation of the angles remains the same
        angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180
        adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
        angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians
        angles_rad = np.append(angles_rad, angles_rad[0])  # Append the start to the end to close the plot

        # Create radial plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)


        # Iterate through the plotting data dictionary to plot both normal and catch data
        for key, data in plotting_data.items():
            if key == 'p_values':
                continue
            performance_data = np.append(data['performance_mean'], data['performance_mean'][0])
            performance_sem = np.append(data['performance_sem'], data['performance_sem'][0])
            performance_sd = np.append(data['performance_sd'], data['performance_sd'][0])

            # Plot the line for this dataset
            ax.plot(angles_rad, performance_data, marker='o', color=colors[key], label=key.capitalize())

            # Add the shaded region for standard deviation or SEM
            if error_bars == 'SD':
                ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, color=lighten_color(colors[key]), alpha=0.4)
            elif error_bars == 'SEM':
                ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, color=lighten_color(colors[key]), alpha=0.4)

        # Adjusting tick labels to reflect left (-) and right (+) turns
        tick_locs = np.radians(np.arange(limits[0], limits[1]+1, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1]+1, 30)]  # Custom labels from -180 to 180

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)

        # Custom plot adjustments
        ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
        ax.set_theta_direction(1)  # Clockwise direction

        # Add text in bottom right
        text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
        ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Add title
        ax.set_title(title, va='bottom', fontsize=16)

        # Add a legend
        ax.legend(loc='upper right')

        sub_dir_name = 'catch_trial_plots'
        # Check if the directory exists in the output path, and create it if it doesn't
        final_output_path = output_path / sub_dir_name  # Create the full path
        if not final_output_path.exists():
            final_output_path.mkdir(parents=True)  # Create the directory, including any missing parents

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # cue_modes_str = '_'.join(cue_modes)  # Join list elements into a string
        output_filename = f"{date_time}_plot_performance_by_angle_radial.svg"

        # Check for existing files and modify filename if necessary
        counter = 0
        while (final_output_path / output_filename).exists():
            output_filename = f"{date_time}_plot_performance_by_angle_radial_{counter}.svg"
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
        text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
        ax.text(0.21, 0.05, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')

        # Add a title
        ax.set_title(title, fontsize=16)

        sub_dir_name = 'catch_trial_plots'
        # Check if the directory exists in the output path, and create it if it doesn't
        final_output_path = output_path / sub_dir_name  # Create the full path
        if not final_output_path.exists():
            final_output_path.mkdir(parents=True)  # Create the directory, including any missing parents

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # cue_modes_str = '_'.join(cue_modes)  # Join list elements into a string
        output_filename = f"{date_time}_plot_performance_by_angle_linear.svg"

        # Check for existing files and modify filename if necessary
        counter = 0
        while (final_output_path / output_filename).exists():
            output_filename = f"{date_time}_plot_performance_by_angle_linear_{counter}.svg"
            counter += 1

        # Save the plot as SVG in the desired folder
        print(f"Saving plot to: '{final_output_path / output_filename}'")
        plt.savefig(final_output_path / output_filename, format='svg', bbox_inches='tight')

        # Show the plot
        plt.tight_layout()
        plt.show()
