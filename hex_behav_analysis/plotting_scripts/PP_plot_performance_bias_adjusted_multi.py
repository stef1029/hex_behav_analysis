from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

def plot_performance_by_angle(sessions_dict, 
                              title='title', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              cue_mode='both',
                              error_bars='SEM',
                              plot_bias=False,
                              output_path=None):
    """
    This function takes a dictionary of session lists and plots the performance by angle for each dataset on the same graph.

    ### Inputs: 
    - sessions_dict: dictionary with keys as labels and values as lists of session objects
    - title: title of the plot
    - bin_mode: 'manual' (->set num_bins), 'rice', 'tpb' (trials per bin ->set tpb value) - to choose the method of binning
    - num_bins: number of bins to divide the angles into
    - trials_per_bin: number of trials per bin for tpb bin mode
    - cue_mode: 'both', 'visual' or 'audio' to choose the type of cue to plot
    - error_bars: 'SEM' or 'SD' for error bars
    - plot_bias: whether to plot raw bias instead of bias-adjusted performance
    - output_path: Path object where to save the output plots
    """
    # Colors list
    colors = [ 
        (0, 0.68, 0.94),  # Blue
        (0.93, 0, 0.55),  # Magenta
        (1, 0.59, 0)      # Orange
    ]

    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    def calc_bias(bins, total_trials):
        """
        Takes the bins which contain how many touches went to each angle, 
        and the total number of trials for that mouse.
        """
        return [sum(bins[key]) / total_trials if bins[key] else 0 for key in sorted(bins)]
        
    # Initialize total_trials list and datasets_data dictionary
    total_trials = []
    datasets_data = {}
    
    for dataset_key, sessions in sessions_dict.items():
        datasets_data[dataset_key] = {'total_trials': [], 'trials': {}, 'performance_data': [], 'bias_data': []}
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')
            if mouse == 'wtjp254-4b':
                continue  # Skip this mouse
            if mouse not in datasets_data[dataset_key]['trials']:
                datasets_data[dataset_key]['trials'][mouse] = []
            # Collect trials based on cue_mode
            if cue_mode == 'both':
                for trial in session.trials:
                    if not trial.get('catch', False):
                        datasets_data[dataset_key]['total_trials'].append(trial)
                        datasets_data[dataset_key]['trials'][mouse].append(trial)
                        total_trials.append(trial)
            elif cue_mode == 'visual':
                for trial in session.trials:
                    if 'audio' not in trial.get('correct_port', '') and not trial.get('catch', False):
                        datasets_data[dataset_key]['total_trials'].append(trial)
                        datasets_data[dataset_key]['trials'][mouse].append(trial)
                        total_trials.append(trial)
            elif cue_mode == 'audio':
                for trial in session.trials:
                    if 'audio' in trial.get('correct_port', '') and not trial.get('catch', False):
                        datasets_data[dataset_key]['total_trials'].append(trial)
                        datasets_data[dataset_key]['trials'][mouse].append(trial)
                        total_trials.append(trial)

    # Determine binning parameters
    n = len(total_trials)
    if bin_mode == 'manual':
        num_bins = num_bins
    elif bin_mode == 'rice':
        num_bins = int(round(2 * n ** (1/3)))
    elif bin_mode == 'tpb':
        num_bins = int(round(n / trials_per_bin))
    else:
        raise ValueError('bin_mode must be "manual", "rice" or "tpb"')
    bin_size = round(360 / num_bins)
    
    # Create bin titles
    bin_titles = [f"{int(i) + (bin_size / 2)}" for i in range(-180, 180, bin_size)]
    
    # For each dataset, process trials per mouse
    for dataset_key in datasets_data:
        performance_data = []
        bias_data = []
        for mouse in datasets_data[dataset_key]['trials']:
            bins = {i: [] for i in range(-180, 180, bin_size)}
            bias = {i: [] for i in range(-180, 180, bin_size)}
            mouse_total_trials = len(datasets_data[dataset_key]['trials'][mouse])
            for trial in datasets_data[dataset_key]['trials'][mouse]:
                if trial["turn_data"] is not None:
                    cue_presentation_angle = trial["turn_data"]["cue_presentation_angle"]
                    port_touched_angle = trial["turn_data"]["port_touched_angle"]
                    if trial["turn_data"]["left_ear_likelihood"] < 0.6:
                        continue
                    if trial["turn_data"]["right_ear_likelihood"] < 0.6:
                        continue
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
            # Compute performance and bias per mouse
            mouse_performance = calc_performance(bins)
            mouse_bias = calc_bias(bias, mouse_total_trials)
            performance_data.append(mouse_performance)
            bias_data.append(mouse_bias)
        # Store performance_data and bias_data for the dataset
        datasets_data[dataset_key]['performance_data'] = np.array(performance_data)
        datasets_data[dataset_key]['bias_data'] = np.array(bias_data)
        # Compute mean and std over mice
        datasets_data[dataset_key]['performance'] = np.mean(datasets_data[dataset_key]['performance_data'], axis=0)
        datasets_data[dataset_key]['performance_sd'] = np.std(datasets_data[dataset_key]['performance_data'], axis=0)
        n_mice = len(performance_data)
        datasets_data[dataset_key]['performance_sem'] = datasets_data[dataset_key]['performance_sd'] / np.sqrt(n_mice)
        datasets_data[dataset_key]['bias'] = np.mean(datasets_data[dataset_key]['bias_data'], axis=0)
        datasets_data[dataset_key]['bias_sd'] = np.std(datasets_data[dataset_key]['bias_data'], axis=0)
        datasets_data[dataset_key]['bias_sem'] = datasets_data[dataset_key]['bias_sd'] / np.sqrt(n_mice)
        # Normalize the bias so that the sum of all biases equals 1
        bias_sum = np.sum(datasets_data[dataset_key]['bias'])
        if bias_sum > 0:
            datasets_data[dataset_key]['bias'] = datasets_data[dataset_key]['bias'] / bias_sum
        # Compute bias-adjusted performance
        epsilon = 1e-10
        biased_performance_data = datasets_data[dataset_key]['performance_data'] / (datasets_data[dataset_key]['bias_data'] + epsilon)
        datasets_data[dataset_key]['biased_performance_data'] = biased_performance_data
        datasets_data[dataset_key]['biased_performance'] = np.mean(biased_performance_data, axis=0)
        datasets_data[dataset_key]['biased_performance_sd'] = np.std(biased_performance_data, axis=0)
        datasets_data[dataset_key]['biased_performance_sem'] = datasets_data[dataset_key]['biased_performance_sd'] / np.sqrt(n_mice)
    
    # Prepare data for plotting
    angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180
    # Adjust angles for plotting and convert to radians
    adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
    angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians
    # Append the start to the end to close the plot
    angles_rad = np.append(angles_rad, angles_rad[0])
    
    # Create radial plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Initialize variables to adjust y-limits dynamically for bias
    if plot_bias:
        all_bias_lower = []
        all_bias_upper = []
    
    # Plotting
    color_index = 0
    for dataset_key in datasets_data:
        color = colors[color_index % len(colors)]
        color_index += 1
    
        if plot_bias:
            # Plot the average raw bias
            bias_data = datasets_data[dataset_key]['bias']
            bias_data = np.append(bias_data, bias_data[0])
            # Convert bias data to percentages
            bias_data_percent = bias_data * 100
            ax.plot(angles_rad, bias_data_percent, marker='o', color=color, label=f'{dataset_key} Bias')
            # Adding the shaded region for error bars
            bias_sem = datasets_data[dataset_key]['bias_sem']
            bias_sem = np.append(bias_sem, bias_sem[0]) * 100  # Convert to percentages
            bias_sd = datasets_data[dataset_key]['bias_sd']
            bias_sd = np.append(bias_sd, bias_sd[0]) * 100  # Convert to percentages
            if error_bars == 'SD':
                lower = bias_data_percent - bias_sd
                upper = bias_data_percent + bias_sd
            elif error_bars == 'SEM':
                lower = bias_data_percent - bias_sem
                upper = bias_data_percent + bias_sem
            else:
                lower = upper = None  # No error bars
            if lower is not None and upper is not None:
                # For radial plots, use fill to create the shaded area
                angles_fill = np.concatenate([angles_rad, angles_rad[::-1]])
                values_fill = np.concatenate([upper, lower[::-1]])
                ax.fill(angles_fill, values_fill, color=color, alpha=0.2)
                # Collect lower and upper values for y-limits adjustment
                all_bias_lower.append(lower)
                all_bias_upper.append(upper)
            else:
                # Collect bias_data_percent for y-limits adjustment
                all_bias_lower.append(bias_data_percent)
                all_bias_upper.append(bias_data_percent)
        else:
            # Plot the average bias-adjusted performance
            biased_performance_data = datasets_data[dataset_key]['biased_performance']
            biased_performance_data = np.append(biased_performance_data, biased_performance_data[0])
            ax.plot(angles_rad, biased_performance_data, marker='o', color=color, label=dataset_key)
            # Adding the shaded region for error bars
            biased_performance_sem = datasets_data[dataset_key]['biased_performance_sem']
            biased_performance_sem = np.append(biased_performance_sem, biased_performance_sem[0])
            biased_performance_sd = datasets_data[dataset_key]['biased_performance_sd']
            biased_performance_sd = np.append(biased_performance_sd, biased_performance_sd[0])
            if error_bars == 'SD':
                lower = biased_performance_data - biased_performance_sd
                upper = biased_performance_data + biased_performance_sd
            elif error_bars == 'SEM':
                lower = biased_performance_data - biased_performance_sem
                upper = biased_performance_data + biased_performance_sem
            else:
                lower = upper = None  # No error bars
            if lower is not None and upper is not None:
                # For radial plots, use fill to create the shaded area
                angles_fill = np.concatenate([angles_rad, angles_rad[::-1]])
                values_fill = np.concatenate([upper, lower[::-1]])
                ax.fill(angles_fill, values_fill, color=color, alpha=0.2)
    
    # Adjusting tick labels to reflect left (-) and right (+) turns
    tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
    tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]  # Custom labels from -180 to 180
    
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    
    # Custom plot adjustments
    ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
    ax.set_theta_direction(1)  # Clockwise direction
    
    # Adjust radial axis labels and limits
    if plot_bias:
        # Concatenate all lower and upper values
        all_bias_lower = np.concatenate(all_bias_lower)
        all_bias_upper = np.concatenate(all_bias_upper)

        # Compute overall min and max
        y_min = np.min(all_bias_lower)
        y_max = np.max(all_bias_upper)

        # Add some margin
        y_range = y_max - y_min
        y_min = max(0, y_min - 0.05 * y_range)
        y_max = min(100, y_max + 0.05 * y_range)

        ax.set_ylim(y_min, y_max)

        # Set radial ticks
        tick_interval = (y_max - y_min) / 5  # 5 intervals
        radial_ticks = np.linspace(y_min, y_max, num=6)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([f"{tick:.0f}%" for tick in radial_ticks])
    else:
        # Optionally adjust radial limits for performance
        pass  # Leave default behavior
    
    # Add text in bottom right with the number of trials and mice
    total_mice = set()
    for dataset_key in datasets_data:
        for mouse in datasets_data[dataset_key]['trials']:
            total_mice.add(mouse)
    text = f"Trials: {len(total_trials)} - Mice: {len(total_mice)}"
    ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
    
    # Add title
    ax.set_title(title, va='bottom', fontsize=16)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Save figures if output_path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
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
    
    # Display the plot
    plt.show()
