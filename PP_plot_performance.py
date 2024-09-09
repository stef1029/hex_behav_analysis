from matplotlib import pyplot as plt
from Session_nwb import Session
from pathlib import Path
from Cohort_folder import Cohort_folder
import json
import numpy as np

# Define your colors
colors = {
    "Cyan": (0, 0.68, 0.94),
    "Magenta": (0.93, 0, 0.55),
    "Orange": (1, 0.59, 0)
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
                              error_bars = 'SEM'):
    """
    This function takes a list of sessions and plots the performance by angle of all trials in the sessions given.
    """
    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]

    # Load trial list:
    total_trials = []
    for session in sessions:
        if cue_mode == 'both':
            for trial in session.trials:
                if not trial.get('catch', False):
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
        mouse = session.session_dict.get('mouse_id', 'unknown')  
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

            for trial in trials[mouse]['trials']:
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

            # caluclate the total lenth of each bin across mice:
            trials[mouse]['performance'] = calc_performance(bins)
            trials[mouse]['n'] = [len(bins[key]) for key in sorted(bins)]
            # print(trials[mouse]['n'])
            bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins)]

        length_data = np.array([trials[mouse]['n'] for mouse in trials])
        length = np.mean(length_data, axis=0)
        # print(length)
        performance_data = np.array([trials[mouse]['performance'] for mouse in trials])
        performance = np.mean(performance_data, axis=0)
        performance_sd = np.std(performance_data, axis=0)
        n = len(performance_data)
        performance_sem = performance_sd / np.sqrt(n)

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
        plt.xlim(0, 180)
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
        plt.xlim(0, 180)
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
        angles_deg = np.array(bin_titles, dtype=np.float64)
        performance_data = np.array(performance)

        adjusted_angles_deg = angles_deg % 360
        angles_rad = np.radians(adjusted_angles_deg)

        angles_rad = np.append(angles_rad, angles_rad[0])
        performance_data = np.append(performance_data, performance_data[0])
        performance_sem = np.append(performance_sem, performance_sem[0])

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles_rad, performance_data, marker='o', color=colors['Cyan'])

        if error_bars == 'SD':
            ax.fill_between(angles_rad, performance_data - performance_sd, performance_data + performance_sd, color=lighten_color(colors['Cyan']), alpha=0.4)
            ax.text(0.9, 0, '±SD', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
        if error_bars == 'SEM':
            ax.fill_between(angles_rad, performance_data - performance_sem, performance_data + performance_sem, color=lighten_color(colors['Cyan']), alpha=0.4)
            ax.text(0.9, 0, '±SEM', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)

        text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
        ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        ax.set_title(title, va='bottom', fontsize=16)
        plt.show()




