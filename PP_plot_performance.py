from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np




def plot_performance_by_angle(sessions, 
                              title = 'title', 
                              bin_mode = 'manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_mode='both',
                              error_bars = 'SEM'):
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
    
    # Load trial list:
    total_trials = []
    for session in sessions:
        if cue_mode == 'both':
            for trial in session.trials:
                if not trial.get('catch', False):  # Check if 'catch' is False or not present
                    total_trials.append(trial)
        elif cue_mode == 'visual':
            for trial in session.trials:
                if 'audio' not in trial.get('correct_port', '') and not trial.get('catch', False):
                    total_trials.append(trial)
        elif cue_mode == 'audio':
            for trial in session.trials:
                if 'audio' in trial.get('correct_port', '') and not trial.get('catch', False):
                    total_trials.append(trial)

    trials = {}
    for session in sessions:
        mouse = session.session_dict.get('mouse_id', 'unknown')  # Use 'unknown' if 'mouse_id' is missing
        if mouse not in trials:
            trials[mouse] = {'trials': []}
        if cue_mode == 'both':
            trials[mouse]['trials'] += [trial for trial in session.trials if not trial.get('catch', False)]
        elif cue_mode == 'visual':
            for trial in session.trials:
                if 'audio' not in trial.get('correct_port', '') and not trial.get('catch', False):
                    trials[mouse]['trials'].append(trial)
        elif cue_mode == 'audio':
            for trial in session.trials:
                if 'audio' in trial.get('correct_port', '') and not trial.get('catch', False):
                    trials[mouse]['trials'].append(trial)


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

    bin_size = round(360 / num_bins)

    bin_titles = []
    performance = []
    
    if plot_mode == 'bar_split' or plot_mode == 'bar_split_overlay':

        for mouse in trials:

            left_bins = {i: [] for i in range(0, 180, bin_size)}
            right_bins = {i: [] for i in range(0, 180, bin_size)}

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
            print(mouse)
            bins = {i: [] for i in range(-180, 180, bin_size)}

            for trial in trials[mouse]['trials']:
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

            trials[mouse]['performance'] = calc_performance(bins)
            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

        performance_data = np.array([trials[mouse]['performance'] for mouse in trials])
        performance = np.mean(performance_data, axis=0)
        performance_sd = np.std(performance_data, axis=0)
        n = len(performance_data)
        performance_sem = performance_sd / np.sqrt(n)

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
        plt.xlim(0, 180)
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

        # Preparation of the data remains the same
        angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180
        performance_data = np.array(performance)  # Assuming performance data is ready

        # Adjust angles for plotting and convert to radians
        adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
        angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians

        # Append the start to the end to close the plot
        angles_rad = np.append(angles_rad, angles_rad[0])
        performance_data = np.append(performance_data, performance_data[0])
        # same for sem and sd:
        performance_sem = np.append(performance_sem, performance_sem[0])
        performance_sd = np.append(performance_sd, performance_sd[0])

        # Create radial plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Polar plot with adjustments
        ax.plot(angles_rad, performance_data, marker='o', color='royalblue')  # Add markers for data points
        # ax.fill(angles_rad, performance_data, alpha=0.25)  # Fill for visual emphasis

        # Adding the shaded region for standard deviation
        if error_bars == 'SD':
            ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, color='skyblue', alpha=0.4)
            # Add note saying it's SD:
            ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
        if error_bars == 'SEM':
            ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, color='skyblue', alpha=0.4)
            ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')


        # Adjusting tick labels to reflect left (-) and right (+) turns
        tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
        tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]  # Custom labels from -180 to 180

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)

        # Custom plot adjustments
        ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
        ax.set_theta_direction(1)  # Clockwise direction

        # add text in bottom right:
        text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
        ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Add title
        ax.set_title(title, va='bottom', fontsize=16)

        # Optionally, save the plot with a specific filename
        # plt.savefig("/cephfs2/srogers/test_output/performance_by_angle_radial.png", dpi=300)
