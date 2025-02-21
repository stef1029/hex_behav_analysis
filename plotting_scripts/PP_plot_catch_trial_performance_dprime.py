from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Define colors for normal and catch trials.
colors = {
    "normal": (0, 0.68, 0.94),
    "catch": (0.93, 0, 0.55)
}

def lighten_color(color, factor=0.5):
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def calc_dprime(hits, total_signal_trials, false_alarms, total_noise_trials, adjustment=0.01):
    """
    Calculate d′ using the adjusted hit and false alarm rates.
    """
    if total_signal_trials == 0 or total_noise_trials == 0:
        return 0
    hit_rate = (hits + adjustment) / (total_signal_trials + 2 * adjustment)
    fa_rate = (false_alarms + adjustment) / (total_noise_trials + 2 * adjustment)
    try:
        dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        return np.clip(dprime, -5, 5)
    except Exception:
        return 0

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
    For each mouse, this function bins trials (by cue_presentation_angle) into discrete bins.
    For each bin, it separately computes d′ for normal trials and catch trials.
    
    The d′ calculation is based on:
      - Counting signal trials in the bin (when the cue falls in that bin),
      - Counting hits (if the trial is correct and the response, as given by port_touched_angle, is in that same bin),
      - Counting noise trials (for bins other than the cue bin) and false alarms (if the response falls there on an incorrect trial).
    
    The resulting d′ values for each group are averaged over mice and plotted on the same graph.
    """
    # Gather total trials (for binning) using the cue_mode filter.
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
    
    # Organize trials by mouse.
    trials_by_mouse = {}
    for session in sessions:
        mouse = session.session_dict['mouse_id']
        if mouse not in trials_by_mouse:
            trials_by_mouse[mouse] = {'trials': []}
        if cue_mode == 'both':
            trials_by_mouse[mouse]['trials'] += session.trials
        elif cue_mode == 'visual':
            for trial in session.trials:
                if 'audio' not in trial['correct_port']:
                    trials_by_mouse[mouse]['trials'].append(trial)
        elif cue_mode == 'audio':
            for trial in session.trials:
                if 'audio' in trial['correct_port']:
                    trials_by_mouse[mouse]['trials'].append(trial)
    
    # Set angle limits.
    if plot_mode in ['linear_comparison', 'bar_split', 'bar_split_overlay']:
        limits = (0, 180)
    else:
        limits = (-180, 180)
    
    angle_range = limits[1] - limits[0]
    n = len(total_trials)
    if bin_mode == 'manual':
        num_bins_actual = num_bins
    elif bin_mode == 'rice':
        num_bins_actual = int(2 * n ** (1/3))
    elif bin_mode == 'tpb':
        num_bins_actual = int(n / trials_per_bin)
    else:
        raise ValueError('bin_mode must be "manual", "rice" or "tpb"')
    
    bin_size = angle_range / num_bins_actual
    bin_edges = np.arange(limits[0], limits[1], bin_size)
    # Create bin labels based on bin centers.
    bin_titles = [f"{(b + bin_size/2):.1f}" for b in bin_edges]
    
    # For each mouse, compute d′ separately for normal and catch trials.
    for mouse in trials_by_mouse:
        # Initialize response data dictionaries for both groups.
        response_data_normal = {b: {'hits': 0, 'total_signal_trials': 0,
                                    'false_alarms': 0, 'total_noise_trials': 0} for b in bin_edges}
        response_data_catch = {b: {'hits': 0, 'total_signal_trials': 0,
                                   'false_alarms': 0, 'total_noise_trials': 0} for b in bin_edges}
        
        for trial in trials_by_mouse[mouse]['trials']:
            if trial["turn_data"] is None:
                continue
            cue_angle = trial["turn_data"].get("cue_presentation_angle")
            response_angle = trial["turn_data"].get("port_touched_angle")
            if cue_angle is None or response_angle is None:
                continue
            
            # Determine which bin the cue and response fall into.
            cue_bin = None
            response_bin = None
            for b in bin_edges:
                if cue_angle >= b and cue_angle < b + bin_size:
                    cue_bin = b
                if response_angle >= b and response_angle < b + bin_size:
                    response_bin = b
            if cue_bin is None or response_bin is None:
                continue
            
            # Determine correctness.
            correct = False
            if trial.get("next_sensor"):
                try:
                    correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                except Exception:
                    correct = False
            
            # Choose the group based on whether the trial is a catch trial.
            if trial.get('catch', False):
                data_dict = response_data_catch
            else:
                data_dict = response_data_normal
            
            # Update counts for the cue bin (signal).
            data_dict[cue_bin]['total_signal_trials'] += 1
            if correct and (cue_bin == response_bin):
                data_dict[cue_bin]['hits'] += 1
            
            # Update counts for noise in bins other than the cue bin.
            for b in data_dict:
                if b != cue_bin:
                    data_dict[b]['total_noise_trials'] += 1
                    if (b == response_bin) and (not correct):
                        data_dict[b]['false_alarms'] += 1
        
        # Compute d′ per bin for the normal group.
        normal_dprime = []
        for b in sorted(response_data_normal.keys()):
            bin_data = response_data_normal[b]
            dprime_val = calc_dprime(bin_data['hits'],
                                     bin_data['total_signal_trials'],
                                     bin_data['false_alarms'],
                                     bin_data['total_noise_trials'])
            normal_dprime.append(dprime_val)
        
        # Compute d′ per bin for the catch group.
        catch_dprime = []
        for b in sorted(response_data_catch.keys()):
            bin_data = response_data_catch[b]
            dprime_val = calc_dprime(bin_data['hits'],
                                     bin_data['total_signal_trials'],
                                     bin_data['false_alarms'],
                                     bin_data['total_noise_trials'])
            catch_dprime.append(dprime_val)
        
        trials_by_mouse[mouse]['normal_dprime'] = normal_dprime
        trials_by_mouse[mouse]['catch_dprime'] = catch_dprime
    
    # Aggregate d′ values across mice.
    mouse_ids = [m for m in trials_by_mouse if 'normal_dprime' in trials_by_mouse[m] and 'catch_dprime' in trials_by_mouse[m]]
    if not mouse_ids:
        raise ValueError("No valid mouse data found for d′ calculation.")
    
    normal_data = np.array([trials_by_mouse[m]['normal_dprime'] for m in mouse_ids])
    catch_data = np.array([trials_by_mouse[m]['catch_dprime'] for m in mouse_ids])
    
    plotting_data = {
        'normal': {
            'dprime_data': normal_data,
            'dprime_mean': np.mean(normal_data, axis=0),
            'dprime_sd': np.std(normal_data, axis=0),
            'dprime_sem': np.std(normal_data, axis=0) / np.sqrt(normal_data.shape[0])
        },
        'catch': {
            'dprime_data': catch_data,
            'dprime_mean': np.mean(catch_data, axis=0),
            'dprime_sd': np.std(catch_data, axis=0),
            'dprime_sem': np.std(catch_data, axis=0) / np.sqrt(catch_data.shape[0])
        },
        'bin_titles': bin_titles
    }
    
    # Helper function to save the figure.
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
        # Convert bin centers to angles.
        angles_deg = np.array([float(t) for t in plotting_data['bin_titles']])
        adjusted_angles_deg = angles_deg % 360
        angles_rad = np.radians(adjusted_angles_deg)
        angles_rad = np.append(angles_rad, angles_rad[0])
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Plot normal d′.
        normal_mean = np.append(plotting_data['normal']['dprime_mean'], plotting_data['normal']['dprime_mean'][0])
        if error_bars == 'SD':
            normal_err = np.append(plotting_data['normal']['dprime_sd'], plotting_data['normal']['dprime_sd'][0])
        elif error_bars == 'SEM':
            normal_err = np.append(plotting_data['normal']['dprime_sem'], plotting_data['normal']['dprime_sem'][0])
        ax.plot(angles_rad, normal_mean, marker='o', color=colors['normal'], label="Normal")
        ax.fill_between(angles_rad, normal_mean - normal_err, normal_mean + normal_err,
                        color=lighten_color(colors['normal']), alpha=0.4)
        
        # Plot catch d′.
        catch_mean = np.append(plotting_data['catch']['dprime_mean'], plotting_data['catch']['dprime_mean'][0])
        if error_bars == 'SD':
            catch_err = np.append(plotting_data['catch']['dprime_sd'], plotting_data['catch']['dprime_sd'][0])
        elif error_bars == 'SEM':
            catch_err = np.append(plotting_data['catch']['dprime_sem'], plotting_data['catch']['dprime_sem'][0])
        ax.plot(angles_rad, catch_mean, marker='o', color=colors['catch'], label="Catch")
        ax.fill_between(angles_rad, catch_mean - catch_err, catch_mean + catch_err,
                        color=lighten_color(colors['catch']), alpha=0.4)
        
        # Dynamically compute y-axis limits based on both curves.
        all_values = []
        for y_vals, err_vals in [(normal_mean, normal_err), (catch_mean, catch_err)]:
            valid_mask = np.isfinite(y_vals)
            if np.any(valid_mask):
                valid_data = y_vals[valid_mask]
                valid_err = err_vals[valid_mask]
                all_values.extend(valid_data)
                all_values.extend(valid_data + valid_err)
                all_values.extend(valid_data - valid_err)
        if all_values:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            # Clip to d' limits of -5 and 5
            y_min = max(y_min, -5)
            y_max = min(y_max, 5)
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_ylim(-1, 1)
        
        # Adjust the polar axes.
        tick_locs = np.radians(np.arange(limits[0], limits[1] + 1, 30)) % (2 * np.pi)
        tick_labels = [f"{int(deg)}" for deg in np.arange(limits[0], limits[1] + 1, 30)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        
        info_text = f"Trials: {len(total_trials)} - Mice: {len(mouse_ids)}"
        ax.text(0.0, 0.0, info_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
        ax.set_title(plot_title, va='bottom', fontsize=16)
        ax.legend(loc='upper right')
        
        save_figure()
        plt.show()
    
    elif plot_mode == 'linear_comparison':
        angles_deg = np.array([float(t) for t in plotting_data['bin_titles']])
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Plot normal d′.
        normal_mean = plotting_data['normal']['dprime_mean']
        if error_bars == 'SD':
            normal_err = plotting_data['normal']['dprime_sd']
        elif error_bars == 'SEM':
            normal_err = plotting_data['normal']['dprime_sem']
        ax.plot(angles_deg, normal_mean, marker='o', color=colors['normal'], label="Normal")
        ax.fill_between(angles_deg, normal_mean - normal_err, normal_mean + normal_err,
                        color=lighten_color(colors['normal']), alpha=0.4)
        
        # Plot catch d′.
        catch_mean = plotting_data['catch']['dprime_mean']
        if error_bars == 'SD':
            catch_err = plotting_data['catch']['dprime_sd']
        elif error_bars == 'SEM':
            catch_err = plotting_data['catch']['dprime_sem']
        ax.plot(angles_deg, catch_mean, marker='o', color=colors['catch'], label="Catch")
        ax.fill_between(angles_deg, catch_mean - catch_err, catch_mean + catch_err,
                        color=lighten_color(colors['catch']), alpha=0.4)
        
        ax.set_xlabel(x_title, fontsize=14)
        ax.set_ylabel(y_title, fontsize=14)
        tick_interval = 30
        nice_ticks = np.arange(limits[0], limits[1] + 1, tick_interval)
        ax.set_xticks(nice_ticks)
        ax.set_xticklabels(nice_ticks, rotation=45)
        ax.set_xlim(limits[0], limits[1])
        
        # Dynamically compute y-axis limits for linear plot.
        all_values = []
        for y_vals, err_vals in [(normal_mean, normal_err), (catch_mean, catch_err)]:
            valid_mask = np.isfinite(y_vals)
            if np.any(valid_mask):
                valid_data = y_vals[valid_mask]
                valid_err = err_vals[valid_mask]
                all_values.extend(valid_data)
                all_values.extend(valid_data + valid_err)
                all_values.extend(valid_data - valid_err)
        if all_values:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            y_min = max(y_min, -5)
            y_max = min(y_max, 5)
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_ylim(-1, 1)
        
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.legend()
        ax.text(0.21, 0.05, f"Trials: {len(total_trials)} - Mice: {len(mouse_ids)}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')
        ax.set_title(plot_title, fontsize=16)
        
        plt.tight_layout()
        save_figure()
        plt.show()
