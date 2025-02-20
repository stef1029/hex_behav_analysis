from matplotlib import pyplot as plt
from pathlib import Path
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
                              plot_title='',
                              x_title='',
                              y_title='',
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              cue_mode='both',
                              error_bars='SEM',
                              output_path=None,
                              plot_save_name='untitled_plot',
                              draft=True):
    """
    Plots performance-by-angle for catch and normal trials from a list of sessions.
    
    The saving logic and some plot modifications are adapted to be similar to another script.
    
    Parameters:
      sessions         : list of session objects
      title            : Title of the plot
      bin_mode         : 'manual', 'rice', or 'tpb' (trials per bin)
      num_bins         : Number of bins (if using 'manual' bin_mode)
      trials_per_bin   : Number of trials per bin (if using 'tpb')
      plot_mode        : 'radial' or 'linear_comparison'
      cue_mode         : 'both', 'visual', or 'audio'
      error_bars       : 'SEM' or 'SD'
      output_path      : Pathlib Path object; if provided, the figure is saved here
      plot_save_name   : Base name for the saved file(s)
      draft            : If True, the filename starts with a timestamp; if False, 'final_' is prepended
      
    Note: This function uses a list of session objects (not session dicts).
    """
    # Helper function for performance calculation
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    # Gather total trials across sessions
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
    
    # Organize trials by mouse
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
    
    # Set angle limits based on plot_mode
    if plot_mode in ['linear_comparison', 'bar_split', 'bar_split_overlay']:
        limits = (0, 180)
        num_bins = 10
    else:
        limits = (-180, 180)
    
    angle_range = limits[1] - limits[0]
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
    
    # Bin trials by angle for each mouse (separately for normal and catch trials)
    for mouse in trials:
        bins_normal = {i: [] for i in range(limits[0], limits[1], bin_size)}
        bins_catch  = {i: [] for i in range(limits[0], limits[1], bin_size)}
        for trial in trials[mouse]['trials']:
            if trial["turn_data"] is not None:
                if plot_mode == 'linear_comparison':
                    angle = abs(trial["turn_data"]["cue_presentation_angle"])
                else:
                    angle = trial["turn_data"]["cue_presentation_angle"]
                for b in bins_normal:
                    if angle < b + bin_size and angle >= b:
                        if trial.get('catch', False):
                            if trial.get("next_sensor", {}):
                                correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                                bins_catch[b].append(1 if correct else 0)
                            else:
                                bins_catch[b].append(0)
                        else:
                            if trial.get("next_sensor", {}):
                                correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                                bins_normal[b].append(1 if correct else 0)
                            else:
                                bins_normal[b].append(0)
        trials[mouse]['normal_performance'] = calc_performance(bins_normal)
        trials[mouse]['catch_performance']  = calc_performance(bins_catch)
        bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(bins_normal)]
    
    # Build plotting data across mice for normal and catch trials
    plotting_data = {}
    normal_data = np.array([trials[mouse]['normal_performance'] for mouse in trials])
    plotting_data['normal'] = {
        'performance_data': normal_data,
        'performance_mean': np.mean(normal_data, axis=0),
        'performance_sd': np.std(normal_data, axis=0),
        'n': len(normal_data),
        'performance_sem': np.std(normal_data, axis=0) / np.sqrt(len(normal_data))
    }
    catch_data = np.array([trials[mouse]['catch_performance'] for mouse in trials])
    plotting_data['catch'] = {
        'performance_data': catch_data,
        'performance_mean': np.mean(catch_data, axis=0),
        'performance_sd': np.std(catch_data, axis=0),
        'n': len(catch_data),
        'performance_sem': np.std(catch_data, axis=0) / np.sqrt(len(catch_data))
    }
    
    # Ensure both datasets have the same number of mice
    assert len(plotting_data['normal']['performance_data']) == len(plotting_data['catch']['performance_data']), "Mismatch in number of mice"
    
    # Perform paired t-tests for each angle bin
    p_values = []
    for bin_index in range(plotting_data['normal']['performance_data'].shape[1]):
        t_stat, p_val = ttest_rel(plotting_data['normal']['performance_data'][:, bin_index],
                                  plotting_data['catch']['performance_data'][:, bin_index])
        p_values.append(p_val)
    p_values = np.array(p_values)
    plotting_data['p_values'] = p_values
    
    # --- Saving helper ---
    def save_figure():
        if output_path is not None:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            if draft:
                base_filename = f"{date_time}_{plot_save_name}"
            else:
                base_filename = f"final_{plot_save_name}"
            output_filename_svg = f"{base_filename}.svg"
            output_filename_png = f"{base_filename}.png"
            counter = 0
            while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
                output_filename_svg = f"{base_filename}_{counter}.svg"
                output_filename_png = f"{base_filename}_{counter}.png"
                counter += 1
            print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
            plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)
            print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
            plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)
    
    # --- Plotting ---
    if plot_mode == 'radial':
        angles_deg = np.array(bin_titles, dtype=np.float64)
        adjusted_angles_deg = angles_deg % 360
        angles_rad = np.radians(adjusted_angles_deg)
        angles_rad = np.append(angles_rad, angles_rad[0])  # Close the circular plot
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Plot normal and catch data with error bands
        for key in ['normal', 'catch']:
            data = plotting_data[key]
            performance_data = np.append(data['performance_mean'], data['performance_mean'][0])
            if error_bars == 'SD':
                error = np.append(data['performance_sd'], data['performance_sd'][0])
            elif error_bars == 'SEM':
                error = np.append(data['performance_sem'], data['performance_sem'][0])
            ax.plot(angles_rad, performance_data, marker='o', color=colors[key], label=key.capitalize())
            ax.fill_between(angles_rad, performance_data - error, performance_data + error,
                            color=lighten_color(colors[key]), alpha=0.4)
        
        # Adjust tick labels for a radial plot
        tick_locs = np.radians(np.arange(limits[0], limits[1]+1, 30)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1]+1, 30)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        
        # Add a text box with trial and mouse count
        info_text = f"Trials: {len(total_trials)} - Mice: {len(trials)}"
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
        ax.set_title(plot_title, va='bottom', fontsize=16)
        ax.legend(loc='upper right')
        
        save_figure()
        plt.show()
    
    elif plot_mode == 'linear_comparison':
        # Use the bin centers for plotting the data, but override the x-axis ticks with “nice” intervals.
        angles_deg = np.array(bin_titles, dtype=np.float64)
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        for key in ['normal', 'catch']:
            data = plotting_data[key]
            performance_data = data['performance_mean']
            if error_bars == 'SD':
                error = data['performance_sd']
            elif error_bars == 'SEM':
                error = data['performance_sem']
            ax.plot(angles_deg, performance_data, marker='o', color=colors[key], label=key.capitalize())
            ax.fill_between(angles_deg, performance_data - error, performance_data + error,
                            color=lighten_color(colors[key]), alpha=0.4)
        
        # Add stars for significant differences based on p-values
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                max_perf = max(plotting_data['normal']['performance_mean'][i],
                               plotting_data['catch']['performance_mean'][i])
                ax.text(angles_deg[i] + 10, max_perf + 0.05, f'* (p={round(p_val, 3)})',
                        fontsize=14, color='black', ha='center')
        
        ax.set_xlabel(x_title, fontsize=14)
        ax.set_ylabel(y_title, fontsize=14)
        # Instead of using bin_titles for ticks, set nice intervals over the full range:
        tick_interval = 30  # adjust as needed
        nice_ticks = np.arange(limits[0], limits[1] + 1, tick_interval)
        ax.set_xticks(nice_ticks)
        ax.set_xticklabels(nice_ticks, rotation=45)
        
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.legend()
        ax.text(0.21, 0.05, f"Trials: {len(total_trials)} - Mice: {len(trials)}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')
        ax.set_title(plot_title, fontsize=16)
        
        plt.tight_layout()
        save_figure()
        plt.show()
