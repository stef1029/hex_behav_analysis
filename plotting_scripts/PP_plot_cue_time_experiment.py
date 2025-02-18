from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

def plot_performance_by_angle(sessions, 
                              cue_times,
                              title='title', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_mode='both',
                              error_bars='SEM',
                              x_label_gap = 30,
                              output_path = None,
                              use_predefined_colors = "yes"):
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

    predefined_colors = {'unlimited': viridis(0.0),  
                        '1000ms': viridis(0.1),    
                        '750ms': viridis(0.2),   
                        '500ms': viridis(0.4),
                        '300ms': viridis(0.5),     
                        '100ms': viridis(0.6),
                        '50ms': viridis(0.7),
                        '25ms': viridis(0.8),
                        '5ms': viridis(1.0)}
    
    marco_defined_colors = {
        "100ms": (0, 0.68, 0.94),   # Cyan
        "50ms": (0.93, 0, 0.55),    # Magenta
        "25ms": (1, 0.59, 0),       # Orange
        "5ms": (0.13, 0.13, 0.67)}  # Dark blue

    def get_colors_for_cue_times(cue_times):
        # Separate 'unlimited' from numeric cue times for sorting
        numeric_cue_times = [cue for cue in cue_times if cue != 'unlimited']
        numeric_cue_times.sort(key=lambda x: float(x.replace('ms', '')))

        # Append 'unlimited' to the end if present
        if 'unlimited' in cue_times:
            sorted_cue_times = numeric_cue_times + ['unlimited']
        else:
            sorted_cue_times = numeric_cue_times

        # Auto-generate colors across the viridis spectrum, in reverse order
        color_dict = {cue: viridis(1 - i / (len(sorted_cue_times) - 1)) for i, cue in enumerate(sorted_cue_times)}
        
        return color_dict


    # Calculate performance for each bin
    def calc_performance(bins):
        # returns a list of the average performance for each bin:
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    def get_trials(session_list):

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


    # Datasets is a dictionary with cue times, and in each cue time is the mice that were used. 
    # In each mouse is the trials. 
    # Later I also add performance data to each mouse. 
    data_sets = {}

    for session_list, cue_time in zip(sessions, cue_times):
        total_trials, trials = get_trials(session_list) # trials is a dict with mouse sorted trials
                                                        # total is all trials in a cue group
        if cue_time not in data_sets:
            data_sets[cue_time] = {'total_trials': [], 'trials': defaultdict(list)}
            # print(f"adding {cue_time}")

        # print(f"Num trials in {cue_time} = {len(total_trials)}")
        data_sets[cue_time]['total_trials'].extend(total_trials)

        # Merging dictionaries in trials
        for mouse, trial_list in trials.items():
            # print(trial_lis/t.keys())
            data_sets[cue_time]['trials'][mouse].extend(trial_list['trials'])

    
    # After the loop, you can flatten `total_trials` and handle merging if necessary
    for cue_time, data in data_sets.items():
        # Flattening `total_trials` into a single list
        # data_sets[cue_time]['total_trials'] = [trial for sublist in data['total_trials'] for trial in sublist]
        data_sets[cue_time]['trials'] = {mouse: {'trials': trial_list} for mouse, trial_list in data['trials'].items()}

    cue_times = list(data_sets.keys())

    if use_predefined_colors == "no":
        predefined_colors = get_colors_for_cue_times(cue_times)

    if use_predefined_colors == "Marco":
        predefined_colors = marco_defined_colors



    for cue_time in data_sets:
        # print(cue_time)
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
        if plot_mode == 'linear_comparison' \
            or plot_mode == 'bar_split' \
                or plot_mode == 'bar_split_overlay' \
                    or plot_mode == 'radial_abs':
            limits = (0, 180)
            num_bins = 10
        else:
            limits = (-180, 180)
        bin_size = round((limits[1] - limits[0]) / num_bins)

        bin_titles = []
        performance = []
    
        # if the plots are for the full range of angles then find performance for the full range:
        for mouse in trials:
            # print(trials[mouse].keys())
            bins = {i: [] for i in range(limits[0], limits[1], bin_size)}
            for trial in trials[mouse]['trials']:
                if trial["turn_data"] != None:
                    if plot_mode == 'radial_abs' or plot_mode == 'linear_comparison':
                        angle = abs(trial["turn_data"]["cue_presentation_angle"])
                    else:
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

        data_sets[cue_time]['performance'] = performance
        data_sets[cue_time]['performance_sem'] = performance_sem
        data_sets[cue_time]['performance_sd'] = performance_sd
        data_sets[cue_time]['bin_titles'] = bin_titles

    # Radial Plot:

    if plot_mode == 'radial' or plot_mode == 'radial_abs':

        # Polar Plot with Adjustments for Consistent Coloring
        plt.figure(figsize=(8, 10))
        ax = plt.subplot(111, polar=True)

        error_fill_color = 'skyblue'  # Shared color for the error fill

        for cue_time in predefined_colors.keys():
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

            label = f"{cue_time} - Trials: {len(total_trials)} - Mice: {len(trials)}"

            color = predefined_colors[cue_time]

            ax.plot(angles_rad, performance_data, marker='o', color=color, label=label)  # Use predefined color
            if error_bars == 'SD':
                ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, color=error_fill_color, alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
            if error_bars == 'SEM':
                ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, color=error_fill_color, alpha=0.4)
                ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        tick_locs = np.radians(np.arange(limits[0], limits[1]+1, x_label_gap)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1]+1, x_label_gap)]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)

        ax.set_theta_zero_location('N')
        if plot_mode == 'radial_abs':
            ax.set_theta_direction(-1)
            ax.set_thetamax(limits[1])
            ax.set_thetamin(limits[0])
        else:
            ax.set_theta_direction(1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1, title='Cue Time:')
        ax.set_title(title, va='bottom', fontsize=16)
        
        # ------ save figures ------    

        # Create directory if it doesn't exist (but don't concatenate path multiple times)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_filename = f"{date_time}_cue_time_comparison_radial"
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

    if plot_mode == 'linear_comparison':

        # Linear Plot with Adjustments for Consistent Coloring
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        error_fill_color = 'skyblue'  # Shared color for the error fill

        for cue_time in predefined_colors.keys():
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

            label = f"{cue_time} - Trials: {len(total_trials)} - Mice: {len(trials)}"
            color = predefined_colors[cue_time]

            ax.plot(angles_deg, performance_data, marker='o', color=color, label=label)  # Use predefined color

            if error_bars == 'SD':
                ax.fill_between(angles_deg, performance_data - performance_sd, performance_data + performance_sd, color=error_fill_color, alpha=0.4)
                ax.text(0.9, -0.05, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

            if error_bars == 'SEM':
                ax.fill_between(angles_deg, performance_data - performance_sem, performance_data + performance_sem, color=error_fill_color, alpha=0.4)
                ax.text(0.9, -0.05, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        ax.set_xlim([limits[0], limits[1]])
        ax.set_ylim(0, 1)

        tick_locs = np.arange(limits[0], limits[1]+1, x_label_gap)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1]+1, x_label_gap)]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)


        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Performance')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=1, title='Cue Time:')

        # ------ save figures ------    

        # Create directory if it doesn't exist (but don't concatenate path multiple times)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_filename = f"{date_time}_cue_time_comparison_line"
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
