from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session

def plot_performance_by_angle(sessions, 
                              title='title', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              cue_mode='both',
                              error_bars='SEM',
                              plot_individual_mice=False,
                              plot_bias=False,
                              output_path=None):
    """
    This function takes a list of sessions and plots the performance by angle of all trials in the sessions given.
    
    ### Inputs: 
    - sessions: list of session objects (Session class already loaded)
    - title: title of the plot
    - bin_mode: 'manual' (->set num_bins), 'rice', 'tpb' (trials per bin ->set tpb value) -  to choose the method of binning
    - num_bins: number of bins to divide the angles into
    - trials_per_bin: number of trials per bin for tbp bin mode
    - cue_mode: 'both', 'visual' or 'audio' to choose the type of cue to plot
    - error_bars: 'SEM' or 'SD' for error bars
    - plot_individual_mice: whether to plot each mouse individually as lines
    - plot_bias: whether to plot raw bias instead of bias-adjusted performance
    """
    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    def calc_bias(bins, total_trials):
        """
        Takes the bins which contain how many touches went to each angle, 
        and the total number of trials for that mouse.
        """
        return [sum(bins[key]) / total_trials if bins[key] else 1 for key in sorted(bins)]

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
        if mouse == 'wtjp254-4b':
            continue
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
    
    for mouse in trials:
        bins = {i: [] for i in range(-180, 180, bin_size)}
        bias = {i: [] for i in range(-180, 180, bin_size)}

        mouse_total_trials = len(trials[mouse]['trials'])
        for trial in trials[mouse]['trials']:
            if trial["turn_data"] is not None:
                cue_presentation_angle = trial["turn_data"]["cue_presentation_angle"]
                port_touched_angle = trial["turn_data"]["port_touched_angle"]
                for bin in bins:
                    # find performance to each angle based on cue presentation angle
                    if cue_presentation_angle < bin + bin_size and cue_presentation_angle >= bin:
                        if trial["next_sensor"] != {}:
                            if int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1]):
                                bins[bin].append(1)
                            else:
                                bins[bin].append(0)
                        else:
                            bins[bin].append(0)
                    # find the number of touches at each angle to see bias:
                    if port_touched_angle is not None:
                        if port_touched_angle < bin + bin_size and port_touched_angle >= bin:
                            bias[bin].append(1)

        trials[mouse]['performance'] = calc_performance(bins)
        trials[mouse]['bias'] = calc_bias(bias, mouse_total_trials)
        bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

    performance_data = np.array([trials[mouse]['performance'] for mouse in trials])
    performance = np.mean(performance_data, axis=0)
    performance_sd = np.std(performance_data, axis=0)
    n = len(performance_data)
    performance_sem = performance_sd / np.sqrt(n)

    bias_data = np.array([trials[mouse]['bias'] for mouse in trials])
    bias = np.mean(bias_data, axis=0)

    # Normalize the bias so that the sum of all biases equals 1
    bias_sum = np.sum(bias)
    if bias_sum > 0:
        bias = bias / bias_sum  # Normalize bias

    bias_sd = np.std(bias_data, axis=0)
    n = len(bias_data)
    bias_sem = bias_sd / np.sqrt(n)

    epsilon = 1e-10
    biased_performance_data = performance_data / (bias_data + epsilon)
    biased_performance = np.mean(biased_performance_data, axis=0)
    biased_performance_sd = np.std(biased_performance_data, axis=0)
    biased_performance_sem = biased_performance_sd / np.sqrt(n)

    # Preparation of the data for plotting remains the same
    angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180

    # Adjust angles for plotting and convert to radians
    adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
    angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians

    # Append the start to the end to close the plot
    angles_rad = np.append(angles_rad, angles_rad[0])

    # Create radial plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    if plot_individual_mice:
        # Plot individual mice performance or bias lines without error bars
        for mouse in trials:
            if plot_bias:
                # Plot raw bias for each mouse
                mouse_bias_data = np.array(trials[mouse]['bias'])
                mouse_bias_data = np.append(mouse_bias_data, mouse_bias_data[0])  # Close the plot
                ax.plot(angles_rad, mouse_bias_data, marker='o', label=f'Mouse {mouse} (Bias)')
            else:
                # Plot bias-adjusted performance for each mouse
                mouse_performance_data = np.array(trials[mouse]['performance'])
                mouse_bias_data = np.array(trials[mouse]['bias'])

                # Bias-adjust the performance data for each mouse
                mouse_biased_performance_data = mouse_performance_data / (mouse_bias_data + epsilon)
                mouse_biased_performance_data = np.append(mouse_biased_performance_data, mouse_biased_performance_data[0])
                ax.plot(angles_rad, mouse_biased_performance_data, marker='o', label=f'Mouse {mouse} (Performance)')

        # Add a legend to differentiate between mice
        # ax.legend(loc='upper right')
    
    else:
        # Plot the average bias or bias-adjusted performance
        if plot_bias:
            # Plot the average raw bias
            bias_data = np.append(bias, bias[0])
            ax.plot(angles_rad, bias_data, marker='o', color='royalblue', label='Average Bias')
        else:
            # Plot the average bias-adjusted performance
            biased_performance_data = np.append(biased_performance, biased_performance[0])
            biased_performance_sem = np.append(biased_performance_sem, biased_performance_sem[0])
            biased_performance_sd = np.append(biased_performance_sd, biased_performance_sd[0])

            ax.plot(angles_rad, biased_performance_data, marker='o', color='royalblue', label='Average Performance')

            # Adding the shaded region for standard deviation or standard error of the mean
            if error_bars == 'SD':
                ax.fill_between(angles_rad, 
                                biased_performance_data - biased_performance_sd, 
                                biased_performance_data + biased_performance_sd, 
                                color='skyblue', alpha=0.4)
                ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

            if error_bars == 'SEM':
                ax.fill_between(angles_rad, 
                                biased_performance_data - biased_performance_sem, 
                                biased_performance_data + biased_performance_sem, 
                                color='skyblue', alpha=0.4)
                ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

    # Adjusting tick labels to reflect left (-) and right (+) turns
    tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
    tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]  # Custom labels from -180 to 180

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)

    # Custom plot adjustments
    ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
    ax.set_theta_direction(1)  # Clockwise direction

    # Add text in bottom right with the number of trials and mice
    text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
    ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

    # Add title
    ax.set_title(title, va='bottom', fontsize=16)

    # ------ save figures ------    

    if output_path is not None:
        # Create directory if it doesn't exist (but don't concatenate path multiple times)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Define the base filename with date and time
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_filename = f"{date_time}_angular_performance_radial"
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
    
    # Display the plot
    plt.show()
