from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
from datetime import datetime
import matplotlib.cm as cm
from pathlib import Path

# Define color scheme (keep existing colors)
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

def lighten_color(color, factor=0.5):
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def calc_dprime(hits, total_signal_trials, false_alarms, total_noise_trials, adjustment=0.01):
    """
    Calculate d' based on hits and false alarms for a specific angle
    """
    if total_signal_trials == 0 or total_noise_trials == 0:
        return 0
        
    # Calculate rates with adjustment to avoid infinities
    hit_rate = (hits + adjustment) / (total_signal_trials + 2*adjustment)
    fa_rate = (false_alarms + adjustment) / (total_noise_trials + 2*adjustment)
    
    # Convert to d'
    try:
        dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        return np.clip(dprime, -5, 5)  # Limit extreme values
    except:
        return 0

def plot_dprime_by_angle(sessions_input, 
                        plot_title='',
                        x_title='',
                        y_title='', 
                        bin_mode='manual', 
                        num_bins=12, 
                        trials_per_bin=10, 
                        plot_mode='radial', 
                        cue_modes=['all_trials'],
                        error_bars='SEM',
                        plot_individual_mice=False,
                        exclusion_mice=[],
                        output_path=None,
                        plot_save_name='untitled_plot',
                        draft=True,
                        likelihood_threshold=0.6):
    """
    Plot d' sensitivity measure across angles.
    """
    def get_trials(sessions):
        mice = {}
        total_trials = {mode: [] for mode in cue_modes}
        
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')
            if mouse in exclusion_mice:
                continue
                
            if mouse not in mice:
                mice[mouse] = {mode: {
                    'trials': [], 
                    'response_data': {}
                } for mode in cue_modes}
            
            for trial in session.trials:
                if trial.get('catch', False):
                    continue
                
                if "all_trials" in cue_modes:
                    mice[mouse]['all_trials']['trials'].append(trial)
                    total_trials['all_trials'].append(trial)
                
                if "visual_trials" in cue_modes and 'audio' not in trial.get('correct_port', ''):
                    mice[mouse]['visual_trials']['trials'].append(trial)
                    total_trials['visual_trials'].append(trial)
                
                if "audio_trials" in cue_modes and 'audio' in trial.get('correct_port', ''):
                    mice[mouse]['audio_trials']['trials'].append(trial)
                    total_trials['audio_trials'].append(trial)
        
        return mice, total_trials

    # Initialize plot and data storage
    y_mins = []
    y_maxs = []
    if plot_mode == 'radial':
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    else:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

    # Handle color selection
    def get_colors(number_of_sessions):
        if number_of_sessions <= 3:
            return [colors['all_trials'], colors['visual_trials'], colors['audio_trials']][:number_of_sessions]
        else:
            cmap = plt.cm.get_cmap('viridis', number_of_sessions)
            return [cmap(i) for i in range(number_of_sessions)]

    # Process input data
    if isinstance(sessions_input, dict):
        if cue_modes != ['all_trials']:
            raise ValueError("When providing a sessions dictionary, cue_modes must be ['all_trials'].")
        sessions_dict = sessions_input
        colors_list = get_colors(len(sessions_dict))
    else:
        sessions_dict = {'Data': sessions_input}
        colors_list = get_colors(1)

    # Set angle limits and calculate bins
    limits = (-180, 180) if plot_mode == 'radial' else (0, 180)
    angle_range = limits[1] - limits[0]
    
    # Process each dataset
    all_plot_data = []  # Store all plot data for y-limit calculation
    for dataset_name, sessions in sessions_dict.items():
        data_sets = {}
        data_sets['mice'], data_sets['total_trials'] = get_trials(sessions)

        n = len(data_sets['total_trials']['all_trials'])
        if n == 0:
            continue

        num_bins_actual = (num_bins if bin_mode == 'manual' 
                          else int(2 * n ** (1/3)) if bin_mode == 'rice'
                          else int(n / trials_per_bin))
        bin_size = angle_range / num_bins_actual

        # Process each cue mode
        for cue_group in cue_modes:
            plotting_data = {
                'dprime': [],
                'dprime_sem': [],
                'n': [],
                'bin_titles': []
            }

            # Process each mouse
            for mouse in data_sets['mice']:
                response_data = {i: {
                    'hits': 0,
                    'total_signal_trials': 0,
                    'false_alarms': 0,
                    'total_noise_trials': 0
                } for i in np.arange(limits[0], limits[1], bin_size)}
                
                trials = data_sets['mice'][mouse][cue_group]['trials']
                if not trials:
                    continue
                
                # Process trials
                for trial in trials:
                    if (trial.get("turn_data") is None or 
                        trial["turn_data"].get("left_ear_likelihood", 1) < likelihood_threshold or
                        trial["turn_data"].get("right_ear_likelihood", 1) < likelihood_threshold):
                        continue

                    if not trial.get("next_sensor"):
                        continue

                    cue_angle = trial["turn_data"]["cue_presentation_angle"]
                    response_angle = trial["turn_data"].get("port_touched_angle")
                    
                    if response_angle is None:
                        continue

                    # Find bins
                    cue_bin = None
                    response_bin = None
                    for bin_start in response_data:
                        if bin_start <= cue_angle < bin_start + bin_size:
                            cue_bin = bin_start
                        if bin_start <= response_angle < bin_start + bin_size:
                            response_bin = bin_start

                    if cue_bin is None or response_bin is None:
                        continue

                    # Update counts
                    correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                    
                    response_data[cue_bin]['total_signal_trials'] += 1
                    if correct and cue_bin == response_bin:
                        response_data[cue_bin]['hits'] += 1
                    
                    for bin_start in response_data:
                        if bin_start != cue_bin:
                            response_data[bin_start]['total_noise_trials'] += 1
                            if bin_start == response_bin and not correct:
                                response_data[bin_start]['false_alarms'] += 1

                # Calculate d' for each bin
                mouse_dprime = []
                for bin_start in sorted(response_data):
                    bin_data = response_data[bin_start]
                    dprime = calc_dprime(
                        bin_data['hits'],
                        bin_data['total_signal_trials'],
                        bin_data['false_alarms'],
                        bin_data['total_noise_trials']
                    )
                    mouse_dprime.append(dprime)

                data_sets['mice'][mouse][cue_group].update({
                    'dprime': mouse_dprime,
                    'response_data': response_data
                })

            # Calculate statistics across mice
            if data_sets['mice']:
                dprime_data = np.array([data_sets['mice'][mouse][cue_group]['dprime'] 
                                      for mouse in data_sets['mice']])
                n_mice = len(data_sets['mice'])
                
                plotting_data.update({
                    'dprime': np.mean(dprime_data, axis=0),
                    'dprime_sem': np.std(dprime_data, axis=0) / np.sqrt(n_mice),
                    'n': n_mice,
                    'bin_titles': [f"{bin + (bin_size / 2)}" for bin in sorted(response_data)]
                })

            # Plot the data
            if plotting_data['bin_titles']:
                angles_deg = np.array(plotting_data['bin_titles'], dtype=np.float64)
                if plot_mode == 'radial':
                    angles_rad = np.radians(angles_deg)
                    angles_rad = np.append(angles_rad, angles_rad[0])

                    plot_data = np.append(plotting_data['dprime'], plotting_data['dprime'][0])
                    error_data = np.append(plotting_data['dprime_sem'], plotting_data['dprime_sem'][0])

                    color = colors_list[list(sessions_dict.keys()).index(dataset_name)]
                    ax.plot(angles_rad, plot_data, marker='o', label=dataset_name, color=color)

                    if error_bars == 'SEM':
                        ax.fill_between(angles_rad, plot_data - error_data, plot_data + error_data,
                                      alpha=0.3, color=lighten_color(color))
                    
                else:  # linear_comparison
                    plot_data = plotting_data['dprime']
                    error_data = plotting_data['dprime_sem']
                    
                    # Store plot data and error bounds for y-limit calculation
                    all_plot_data.append({
                        'data': plot_data,
                        'error': error_data if error_bars == 'SEM' else None
                    })

                    color = colors_list[list(sessions_dict.keys()).index(dataset_name)]
                    ax.plot(angles_deg, plot_data, marker='o', label=dataset_name, color=color)

                    if error_bars == 'SEM':
                        ax.fill_between(angles_deg, plot_data - error_data, plot_data + error_data,
                                      alpha=0.3, color=lighten_color(color))

    # Final plot adjustments
    ax.legend(loc='upper right')
    ax.set_title(plot_title)
    
    if plot_mode == 'radial':
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        
        angles_label = np.arange(limits[0], limits[1], bin_size)
        ax.set_xticks(np.radians(angles_label))
        ax.set_xticklabels([f'{int(angle)}°' for angle in angles_label])
        
        ax.set_thetamin(-180)
        ax.set_thetamax(180)
        ax.grid(True)
        
        if all_plot_data:
            all_values = []
            for plot_dict in all_plot_data:
                data = plot_dict['data']
                error = plot_dict['error']
                
                valid_mask = np.isfinite(data)
                if np.any(valid_mask):
                    valid_data = data[valid_mask]
                    all_values.extend(valid_data)
                    
                    if error is not None:
                        valid_error = error[valid_mask]
                        all_values.extend(valid_data + valid_error)
                        all_values.extend(valid_data - valid_error)

            if all_values:
                y_min = np.min(all_values)
                y_max = np.max(all_values)
                
                # Ensure we don't exceed d' limits of -5 to 5
                y_min = max(y_min, -5)
                y_max = min(y_max, 5)
                
                # Add padding and set limits
                padding = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - padding, y_max + padding)
            else:
                ax.set_ylim(-1, 1)
    
    else:  # linear_comparison
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_xlim(limits[0], limits[1])
        
        # Calculate overall y-limits from all datasets
        all_values = []
        for plot_dict in all_plot_data:
            data = plot_dict['data']
            error = plot_dict['error']
            
            valid_mask = np.isfinite(data)
            if np.any(valid_mask):
                valid_data = data[valid_mask]
                all_values.extend(valid_data)
                
                if error is not None:
                    valid_error = error[valid_mask]
                    all_values.extend(valid_data + valid_error)
                    all_values.extend(valid_data - valid_error)

        if all_values:
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            
            # Ensure we don't exceed d' limits of -5 to 5
            y_min = max(y_min, -5)
            y_max = min(y_max, 5)
            
            # Add padding and set limits
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_ylim(-1, 1)
            
        ax.grid(True)
        
        # Add axis lines
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        
        # Make top and right spines invisible
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust tick positions
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    # Save plot if output path provided
    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cue_modes_str = '_'.join(cue_modes)
        
        base_filename = f"{date_time if draft else 'final'}_{plot_save_name}_{cue_modes_str}"
        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"

        counter_svg = 0
        counter_png = 0
        while (output_path / output_filename_svg).exists():
            output_filename_svg = f"{base_filename}_{counter_svg}.svg"
            counter_svg += 1
        while (output_path / output_filename_png).exists():
            output_filename_png = f"{base_filename}_{counter_png}.png"
            counter_png += 1

        print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
        plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

        # Save the plot as PNG in the desired folder
        print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
        plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)
    
    # Display the plot
    plt.show()