from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np




def plot_performance_by_angle(sessions, 
                              cue_times,
                              title='title', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_mode='both',
                              error_bars='SEM',
                              x_label_gap = 30):
    """
    This function takes a list of lists of sessions and plots the performance by angle of all trials in the sessions given.
    ### Inputs: 
    - sessions: list of session objects (Session class already loaded)
    - cue_times: list of cue time labels corresponding to each session list
    - title: title of the plot
    - bin_mode: 'manual' (->set num_bins), 'rice', 'tpb' (trials per bin ->set tpb value) -  to choose the method of binning
    - num_bins: number of bins to divide the angles into
    - trials_per_bin: number of trials per bin for tbp bin mode.
    - plot_mode: 'radial' or 'bar' to choose the type of plot
    - cue_mode: 'both', 'visual' or 'audio' to choose the type of cue to plot
    """
    # Define a color map
    viridis = plt.cm.get_cmap('viridis')
    # predefined_colors = {
    #     'Unlimited': viridis(0.0),
    #     '1000ms': viridis(0.25),
    #     '500ms': viridis(0.5),
    #     '300ms': viridis(0.75),
    #     '100ms': viridis(1.0)
    # }

    predefined_colors = {
        '750ms': viridis(0.0),
        '500ms': viridis(0.2),
        '100ms': viridis(0.4),
        '50ms': viridis(0.6),
        '25ms': viridis(0.8),
        '5ms': viridis(1.0)
    }


    # Calculate performance for each bin
    def calc_performance(bins):
        # returns a list of the average performance for each bin:
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    def get_trials(session_list):
        # Load trial list:
        # total_trials = []
        # for session in session_list:
        #     if cue_mode == 'both':
        #         total_trials += session.trials
        #     elif cue_mode == 'visual':
        #         for trial in session.trials:
        #             if 'audio' not in trial['correct_port']:
        #                 total_trials.append(trial)
        #     elif cue_mode == 'audio':
        #         for trial in session.trials:
        #             if 'audio' in trial['correct_port']:
        #                 total_trials.append(trial)

        trials = {}
        total_trials = []
        for session in session_list:
            mouse = session.session_dict['mouse_id']
            if mouse not in trials:
                trials[mouse] = {'trials': []}
            if cue_mode == 'both':
                trials[mouse]['trials'] += session.trials
                total_trials += session.trials
            elif cue_mode == 'visual':
                for trial in session.trials:
                    if 'audio' not in trial['correct_port']:
                        trials[mouse]['trials'].append(trial)
                        total_trials.append(trial)
            elif cue_mode == 'audio':
                for trial in session.trials:
                    if 'audio' in trial['correct_port']:
                        trials[mouse]['trials'].append(trial)
                        total_trials.append(trial)
        
        return total_trials, trials

    data_sets = {}
    # cue_times = ['Unlimited', '1000ms', '500ms', '300ms', '100ms']

    data_sets = {}

    for session_list, cue_time in zip(sessions, cue_times):
        total_trials, trials = get_trials(session_list)
        if cue_time not in data_sets:
            data_sets[cue_time] = {}
        data_sets[cue_time]['total_trials'] = total_trials
        data_sets[cue_time]['trials'] = trials

    for cue_time in data_sets:
        # trials is a dict of mouse names and the trials that I'm interested in (all, visual or audio).
        trials = data_sets[cue_time]['trials']
        # bin the trials into 30 degree bins, ranging from 180 to -180
        # get total n trials across all mice:
        n = len(data_sets[cue_time]['total_trials'])

        # determine num bins and then bin size:
        if bin_mode == 'manual':
            num_bins = num_bins
        elif bin_mode == 'rice':
            num_bins = 2 * n ** (1/3)
        elif bin_mode == 'tpb':
            num_bins = n / trials_per_bin
        else:
            raise ValueError('bin_mode must be "manual", "rice" or "tpb"')
        
        # based on the range of angles in use, find the bin size baseed on num of bins:
        if plot_mode == 'radial_abs':
            angle_range = 180
        else:
            angle_range = 360
        bin_size = round(angle_range / num_bins)

        bin_titles = []
        performance = []
        
        # if using the split bar charts, left and right turns are split from each other and so two different plots are created:
        if plot_mode == 'bar_split' or plot_mode == 'bar_split_overlay':
            
            # for each mouse find the data set and then average later so I can find the SEM:
            for mouse in trials:
                angle_range = angle_range / 2
                left_bins = {i: [] for i in range(0, angle_range, bin_size)}
                right_bins = {i: [] for i in range(0, angle_range, bin_size)}

                # Bin trials based on turn direction and angle:
                # for each trial, get the cue_presentation_angle and use that to put the trial in the correct bin:
                for trial in trials[mouse]['trials']:
                    if trial["turn_data"] is not None:
                        # get angle:
                        angle = trial["turn_data"]["cue_presentation_angle"]
                        # check that there was a sensor touch:
                        if trial["next_sensor"] != {}:
                            # calculate if the mouse touched the right port:
                            correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                        else:
                            # if the mouse didn't touch a sensor then it was an error:
                            correct = 0
                            
                        if angle < 0:  # Left turn
                            bin_index = abs(angle) // bin_size * bin_size
                            left_bins[bin_index].append(correct)
                        elif angle > 0:  # Right turn
                            bin_index = angle // bin_size * bin_size
                            right_bins[bin_index].append(correct)

                # make lists of performance values for each bin:
                trials[mouse]['left_performance'] = calc_performance(left_bins)
                trials[mouse]['right_performance'] = calc_performance(right_bins)
                bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(left_bins)] 

            # calculate statistics for each plot:
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

            # save that data to the data_sets dict for later use:
            data_sets[cue_time]['left_performance'] = left_performance
            data_sets[cue_time]['left_performance_sem'] = left_performance_sem
            data_sets[cue_time]['left_performance_sd'] = left_performance_sd
            data_sets[cue_time]['right_performance'] = right_performance
            data_sets[cue_time]['right_performance_sem'] = right_performance_sem
            data_sets[cue_time]['right_performance_sd'] = right_performance_sd
            data_sets[cue_time]['bin_titles'] = bin_titles

        elif plot_mode == 'radial_abs':
            
            for mouse in trials:
                bins = {i: [] for i in range(0, 180, bin_size)}

                for trial in trials[mouse]['trials']:
                    if trial["turn_data"] != None:
                        angle = abs(trial["turn_data"]["cue_presentation_angle"])
                        for bin in bins:
                            if angle < bin + bin_size and angle >= bin:
                                if trial["next_sensor"] != {}:
                                    if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                        bins[bin].append(1)
                                    else:
                                        bins[bin].append(0)

                trials[mouse]['performance'] = calc_performance(bins)
                bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

            performance_data = np.array([trials[mouse]['performance'] for mouse in trials])
            performance = np.mean(performance_data, axis=0)
            performance_sd = np.std(performance_data, axis=0)
            n = len(performance_data)
            performance_sem = performance_sd / np.sqrt(n)

            data_sets[cue_time]['performance'] = performance
            data_sets[cue_time]['performance_sem'] = performance_sem
            data_sets[cue_time]['performance_sd'] = performance_sd
            data_sets[cue_time]['bin_titles'] = bin_titles

        else:
            # if the plots are for the full range of angles then find performance for the full range:
            for mouse in trials:
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

                trials[mouse]['performance'] = calc_performance(bins)
                bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

            performance_data = np.array([trials[mouse]['performance'] for mouse in trials])
            performance = np.mean(performance_data, axis=0)
            performance_sd = np.std(performance_data, axis=0)
            n = len(performance_data)
            performance_sem = performance_sd / np.sqrt(n)

            data_sets[cue_time]['performance'] = performance
            data_sets[cue_time]['performance_sem'] = performance_sem
            data_sets[cue_time]['performance_sd'] = performance_sd
            data_sets[cue_time]['bin_titles'] = bin_titles

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

    if plot_mode == 'radial' or plot_mode == 'radial_abs':

        # Polar Plot with Adjustments for Consistent Coloring
        plt.figure(figsize=(8, 10))
        ax = plt.subplot(111, polar=True)

        error_fill_color = 'skyblue'  # Shared color for the error fill

        for cue_time in cue_times:
            if cue_time not in data_sets:
                continue
            bin_titles = data_sets[cue_time]['bin_titles']
            performance = data_sets[cue_time]['performance']
            performance_sem = data_sets[cue_time]['performance_sem']
            performance_sd = data_sets[cue_time]['performance_sd']
            total_trials = data_sets[cue_time]['total_trials']
            trials = data_sets[cue_time]['trials']

            angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180
            performance_data = np.array(performance)  # Assuming performance data is ready

            adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
            angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians

            if plot_mode != 'radial_abs':
                angles_rad = np.append(angles_rad, angles_rad[0])
                performance_data = np.append(performance_data, performance_data[0])
                performance_sem = np.append(performance_sem, performance_sem[0])
                performance_sd = np.append(performance_sd, performance_sd[0])

            label = f"{cue_time} - Trials: {len(total_trials)} - Mice: {len(trials)} - Average trials/ bin: {round(len(total_trials)/len(bin_titles))}"

            color = predefined_colors[cue_time]

            ax.plot(angles_rad, performance_data, marker='o', color=color, label=label)  # Use predefined color
            if error_bars == 'SD':
                ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, color=error_fill_color, alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            if error_bars == 'SEM':
                ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, color=error_fill_color, alpha=0.4)
                ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        tick_locs = np.radians(np.arange(-180, 181, x_label_gap)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, x_label_gap)]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)

        ax.set_theta_zero_location('N')
        if plot_mode == 'radial_abs':
            ax.set_theta_direction(-1)
            ax.set_thetamax(180)
            ax.set_thetamin(0)
        else:
            ax.set_theta_direction(1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1, title='Cue Time:')
        ax.set_title(title, va='bottom', fontsize=16)
