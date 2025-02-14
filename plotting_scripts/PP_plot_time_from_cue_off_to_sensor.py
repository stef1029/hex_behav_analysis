from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from matplotlib.patches import Circle

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session


def plot_performance_by_angle(sessions, 
                              title = 'title', 
                              bin_mode = 'manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_mode='both',
                              error_bars = 'SEM',
                              time_to_plot = 'time_after',
                              trial_type = 'all',
                              cue_duration = 0):
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
    - error_bars: 'SEM' or 'SD' to choose the type of error bars to plot
    - time_to_plot: 'time_after'(default) or 'total_time' to choose time to use.
    - trial_type: 'all', 'success', 'failure', 'overlay' to choose the type of trials to plot (overlay = success and fail overlayed)
    """
    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    # Load trial list:
    # Load all trials into one list
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

    # Load trials into mouse specific lists:
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


    # bin the trials into bins, ranging from 180 to -180
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

            left_bins = {i: {'total': [], 'success': [], 'failure': []} for i in range(0, 180, bin_size)}
            right_bins = {i: {'total': [], 'success': [], 'failure': []} for i in range(0, 180, bin_size)}

            # Bin trials based on turn direction, angle, and success
            for trial in trials[mouse]['trials']:
                if trial["turn_data"] is not None:
                    angle = trial["turn_data"]["cue_presentation_angle"]
                    correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                    total_trial_time = float(trial['next_sensor']['sensor_start']) - float(trial['cue_start'])
                    time_after_cue = float(trial['next_sensor']['sensor_start']) - float(trial['cue_end'])
                    if time_after_cue < 0:
                        time_after_cue = 0

                    if angle < 0:  # Left turn
                        bin_index = abs(angle) // bin_size * bin_size
                        left_bins[bin_index]['total'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)
                        if correct:
                            left_bins[bin_index]['success'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)
                        else:
                            left_bins[bin_index]['failure'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)

                    elif angle > 0:  # Right turn
                        bin_index = angle // bin_size * bin_size
                        right_bins[bin_index]['total'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)
                        if correct:
                            right_bins[bin_index]['success'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)
                        else:
                            right_bins[bin_index]['failure'].append(time_after_cue if time_to_plot == 'time_after' else total_trial_time)

            # Calculate average times for each bin
            for bins in [left_bins, right_bins]:
                for key in bins:
                    if bins[key]['total']:
                        bins[key]['total'] = np.mean(bins[key]['total'])
                        bins[key]['success'] = np.mean(bins[key]['success']) if bins[key]['success'] else 0
                        bins[key]['failure'] = np.mean(bins[key]['failure']) if bins[key]['failure'] else 0

        # Extract and plot data
        left_performance = [left_bins[key]['total'] for key in sorted(left_bins)]
        right_performance = [right_bins[key]['total'] for key in sorted(right_bins)]


    else:
        for mouse in trials:
            bins = {i: {'total': [], 'success': [], 'failure': []} for i in range(-180, 180, bin_size)}

            for trial in trials[mouse]['trials']:
                if trial["turn_data"] != None:
                    if trial["next_sensor"] != {}:
                        angle = trial["turn_data"]["cue_presentation_angle"]
                        success = 1 if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]) else 0
                        total_trial_time = float(trial['next_sensor']['sensor_start']) - float(trial['cue_start']) 
                        time_after_cue = float(trial['next_sensor']['sensor_start']) - float(trial['cue_end'])
                        if time_after_cue < 0:
                            time_after_cue = 0
                        
                        # For each bin, make a list of times, for the total trials, 
                        #    and then also split into success and failure
                        for bin in bins:
                            if angle < bin + bin_size and angle >= bin:
                                if time_to_plot == 'time_after':
                                    bins[bin]['total'].append(time_after_cue)
                                    # print(time_after_cue)
                                    if success == 1:
                                        bins[bin]['success'].append(time_after_cue)
                                    else:
                                        bins[bin]['failure'].append(time_after_cue)
                                elif time_to_plot == 'total_time':
                                    bins[bin]['total'].append(total_trial_time)
                                    if success == 1:
                                        bins[bin]['success'].append(total_trial_time)
                                    else:
                                        bins[bin]['failure'].append(total_trial_time)
                                else:
                                    raise ValueError('time_to_plot must be "time_after" or "total_time"')
            
            if 'times' not in trials[mouse]:
                trials[mouse]['times'] = {}
            trials[mouse]['times']['total'] = [np.mean(bins[bin]['total']) for bin in bins]
            trials[mouse]['times']['success'] = [np.mean(bins[bin]['success']) for bin in bins]
            trials[mouse]['times']['failure'] = [np.mean(bins[bin]['failure']) for bin in bins]

            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

        total_times_data = np.array([trials[mouse]['times']['total'] for mouse in trials])
        total_average_times = np.mean(total_times_data, axis=0)
        total_times_sd = np.std(total_times_data, axis=0)
        total_n = len(total_times_data)
        total_times_sem = total_times_sd / np.sqrt(n)

        success_times_data = np.array([trials[mouse]['times']['success'] for mouse in trials])
        success_average_times = np.mean(success_times_data, axis=0)
        success_times_sd = np.std(success_times_data, axis=0)
        success_n = len(success_times_data)
        success_times_sem = success_times_sd / np.sqrt(n)

        failure_times_data = np.array([trials[mouse]['times']['failure'] for mouse in trials])
        failure_average_times = np.mean(failure_times_data, axis=0)
        failure_times_sd = np.std(failure_times_data, axis=0)
        failure_n = len(failure_times_data)
        failure_times_sem = failure_times_sd / np.sqrt(n)



    def plot_performance(bin_titles, times, errors, title, color_map='viridis'):
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

    def plot_performance_multi(bin_titles, left_times, left_errors, right_times, right_errors, left_title, right_title, color_map='viridis'):
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

        if trial_type == 'all':
            times = [total_average_times]
            sds = [total_times_sd]
            sems = [total_times_sem]
            line_label = ['All Trials']
        elif trial_type == 'success':
            times = [success_average_times]
            sds = [success_times_sd]
            sems = [success_times_sem]
            line_label = ['Successful Trials']
        elif trial_type == 'failure':
            times = [failure_average_times]
            sds = [failure_times_sd]
            sems = [failure_times_sem]
            line_label = ['Failed Trials']
        elif trial_type == 'overlay':
            times = [success_average_times, failure_average_times]
            sds = [success_times_sd, failure_times_sd]
            sems = [success_times_sem, failure_times_sem]
            line_label = ['Successful Trials', 'Unsuccessful Trials']
        else:
            raise ValueError('trial_type must be "all", "success", "failure", or "overlay"')

        # Adjust angles for plotting and convert to radians
        adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
        angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians

        # Append the start to the end to close the plot
        angles_rad = np.append(angles_rad, angles_rad[0])
        times = [np.append(time, time[0]) for time in times]
        # same for sem and sd:
        times_sem = [np.append(sem, sem[0]) for sem in sems]
        times_sd = [np.append(sd, sd[0]) for sd in sds]

        # Create radial plot
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Polar plot with adjustments
        for time, time_sem, time_sd, label in zip(times, times_sem, times_sd, line_label):
            if error_bars == 'SD':
                ax.fill_between(angles_rad, time - time_sd, time + time_sd, color='skyblue', alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            elif error_bars == 'SEM':
                ax.fill_between(angles_rad, time - time_sem, time + time_sem, color='skyblue', alpha=0.4)
            else:
                raise ValueError('error_bars must be "SD" or "SEM"')
            ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            ax.plot(angles_rad, time, marker='o', label = label)


        # Adjusting tick labels to reflect left (-) and right (+) turns
        tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
        tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]  # Custom labels from -180 to 180

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)

        # get max duratation of the trials to set the radius of the shaded circle
        max_durations = [np.nanmax(time) if np.any(~np.isnan(time)) else None for time in times]  # Handles arrays that are all NaN
        max_duration = np.nanmax(max_durations)  # This will ignore None because np.nanmax handles NaN/None gracefully in a numeric context

        y_lim = round(max_duration + 0.5)
        ax.set_ylim(0, y_lim)
        if cue_duration == 'unlimited':
            cue_duration = y_lim

        # Add shaded circle at the center of the polar plot
        circle = Circle((0, 0), cue_duration, transform=ax.transData._b, color='green', alpha=0.3, linewidth=0)
        ax.add_artist(circle) 

        from matplotlib.lines import Line2D
        legend_circle = Line2D([0], [0], marker='o', color='w', label='Target Area',
                            markerfacecolor='red', markersize=10)
        ax.legend(handles=[legend_circle])      

        # Custom plot adjustments
        ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
        ax.set_theta_direction(1)  # Clockwise direction

        # add text in bottom right:
        text = f"Trials: {len(total_trials)} - Mice: {len(sessions)}"
        ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Add title
        ax.set_title(title, va='bottom', fontsize=16)
        ax.legend()

        # Optionally, save the plot with a specific filename
        # plt.savefig("/cephfs2/srogers/test_output/performance_by_angle_radial.png", dpi=300)
