from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.cm as cm

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

# Define your colors
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

# Function to lighten a color for shaded regions
def lighten_color(color, factor=0.5):
    return tuple(min(1, c + (1 - c) * factor) for c in color)

def plot_performance_by_angle(sessions_input, 
                              plot_title='title',
                              x_title='',
                              y_title='', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              plot_type='hit_rate',  # New parameter: 'hit_rate', 'bias_corrected', or 'bias'
                              cue_modes=['all_trials'],
                              error_bars='SEM',
                              plot_individual_mice=False,
                              exclusion_mice=[],
                              output_path=None,
                              plot_save_name='untitled_plot',
                              draft=True,
                              likelihood_threshold=0.6):  # Added likelihood threshold parameter
    """
    This function takes a list of sessions or a dictionary of session lists and plots angular performance data.
    
    Parameters:
    -----------
    plot_type : str
        Type of plot to generate:
        - 'hit_rate': Raw performance/hit rate
        - 'bias_corrected': Bias-corrected performance
        - 'bias': Raw response bias
    likelihood_threshold : float
        Threshold for ear detection likelihood (default: 0.6)
    """
    
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]
    
    def calc_bias(bins, total_trials):
        """Calculate bias as proportion of touches to each angle"""
        bias_values = [sum(bins[key]) / total_trials if bins[key] else 0 for key in sorted(bins)]
        # Normalize so sum equals 1
        total = sum(bias_values)
        return [v/total if total > 0 else 0 for v in bias_values]

    def get_trials(sessions):
        mice = {}
        total_trials = {mode: [] for mode in cue_modes}
        
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')
            if mouse in exclusion_mice:
                continue
                
            if mouse not in mice:
                mice[mouse] = {mode: {'trials': [], 'bins': {}, 'bias_bins': {}} for mode in cue_modes}
            
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

    # Initialize plot
    if plot_mode == 'radial':
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    else:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

    # Handle color selection for multiple datasets
    def get_colors(number_of_sessions):
        if number_of_sessions <= 3:
            return [colors['all_trials'], colors['visual_trials'], colors['audio_trials']][:number_of_sessions]
        else:
            cmap = plt.cm.get_cmap('viridis', number_of_sessions)
            return [cmap(i) for i in range(number_of_sessions)]

    # Process data
    if isinstance(sessions_input, dict):
        if cue_modes != ['all_trials']:
            raise ValueError("When providing a sessions dictionary, cue_modes must be ['all_trials'].")
        sessions_dict = sessions_input
        colors_list = get_colors(len(sessions_dict))
    else:
        sessions_dict = {'Data': sessions_input}
        colors_list = get_colors(1)

    # Calculate bin parameters
    for dataset_name, sessions in sessions_dict.items():
        data_sets = {}
        data_sets['mice'], data_sets['total_trials'] = get_trials(sessions)

        n = len(data_sets['total_trials']['all_trials'])
        if bin_mode == 'manual':
            num_bins_used = num_bins
        elif bin_mode == 'rice':
            num_bins_used = int(2 * n ** (1/3))
        elif bin_mode == 'tpb':
            num_bins_used = int(n / trials_per_bin)
        else:
            raise ValueError('bin_mode must be "manual", "rice" or "tpb"')

        # Set angle limits based on plot mode
        if plot_mode in ['linear_comparison', 'bar_split', 'bar_split_overlay']:
            limits = (0, 180)
            num_bins_used = 6
        else:
            limits = (-180, 180)

        angle_range = limits[1] - limits[0]
        bin_size = angle_range / num_bins_used

        # Process each cue mode
        for cue_group in cue_modes:
            plotting_data = {
                'performance': [],
                'performance_sd': [],
                'performance_sem': [],
                'bias': [],
                'bias_sd': [],
                'bias_sem': [],
                'bias_corrected': [],
                'bias_corrected_sd': [],
                'bias_corrected_sem': [],
                'n': [],
                'bin_titles': []
            }

            # Process each mouse
            for mouse in data_sets['mice']:
                perf_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}
                bias_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}
                
                for trial in data_sets['mice'][mouse][cue_group]['trials']:
                    if trial.get("turn_data") is None:
                        continue
                        
                    # Check ear detection likelihood
                    if (trial["turn_data"].get("left_ear_likelihood", 1) < likelihood_threshold or
                        trial["turn_data"].get("right_ear_likelihood", 1) < likelihood_threshold):
                        continue

                    angle = trial["turn_data"]["cue_presentation_angle"]
                    port_touched_angle = trial["turn_data"].get("port_touched_angle")

                    for bin in perf_bins:
                        if bin <= angle < bin + bin_size:
                            # Performance calculation
                            if trial.get("next_sensor"):
                                correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                                perf_bins[bin].append(1 if correct else 0)
                            else:
                                perf_bins[bin].append(0)
                            
                            # Bias calculation
                            if port_touched_angle is not None:
                                if bin <= port_touched_angle < bin + bin_size:
                                    bias_bins[bin].append(1)

                # Calculate metrics for this mouse
                mouse_performance = calc_performance(perf_bins)
                mouse_bias = calc_bias(bias_bins, len(data_sets['mice'][mouse][cue_group]['trials']))
                
                # Store results
                data_sets['mice'][mouse][cue_group].update({
                    'performance': mouse_performance,
                    'bias': mouse_bias,
                    'bias_corrected': np.array(mouse_performance) / (np.array(mouse_bias) + 1e-10),
                    'n': [len(perf_bins[key]) for key in sorted(perf_bins)]
                })

            # Calculate statistics across mice
            performance_data = np.array([data_sets['mice'][mouse][cue_group]['performance'] for mouse in data_sets['mice']])
            bias_data = np.array([data_sets['mice'][mouse][cue_group]['bias'] for mouse in data_sets['mice']])
            bias_corrected_data = np.array([data_sets['mice'][mouse][cue_group]['bias_corrected'] for mouse in data_sets['mice']])

            n_mice = len(data_sets['mice'])
            
            # Update plotting data with calculated statistics
            plotting_data.update({
                'performance': np.mean(performance_data, axis=0),
                'performance_sd': np.std(performance_data, axis=0),
                'performance_sem': np.std(performance_data, axis=0) / np.sqrt(n_mice),
                'bias': np.mean(bias_data, axis=0),
                'bias_sd': np.std(bias_data, axis=0),
                'bias_sem': np.std(bias_data, axis=0) / np.sqrt(n_mice),
                'bias_corrected': np.mean(bias_corrected_data, axis=0),
                'bias_corrected_sd': np.std(bias_corrected_data, axis=0),
                'bias_corrected_sem': np.std(bias_corrected_data, axis=0) / np.sqrt(n_mice),
                'n': n_mice,
                'bin_titles': [f"{bin + (bin_size / 2)}" for bin in sorted(perf_bins)]
            })

            # Plot the data
            angles_deg = np.array(plotting_data['bin_titles'], dtype=np.float64)
            if plot_mode == 'radial':
                # Ensure the plot closes by appending first value to end
                adjusted_angles_deg = angles_deg % 360
                angles_rad = np.radians(adjusted_angles_deg)
                angles_rad = np.append(angles_rad, angles_rad[0])

                # Select data based on plot_type and ensure circular closure
                if plot_type == 'hit_rate':
                    plot_data = np.append(plotting_data['performance'], plotting_data['performance'][0])
                    error_data = np.append(plotting_data['performance_sem'], plotting_data['performance_sem'][0])
                elif plot_type == 'bias_corrected':
                    plot_data = np.append(plotting_data['bias_corrected'], plotting_data['bias_corrected'][0])
                    error_data = np.append(plotting_data['bias_corrected_sem'], plotting_data['bias_corrected_sem'][0])
                else:  # bias
                    plot_data = np.append(plotting_data['bias'], plotting_data['bias'][0])
                    error_data = np.append(plotting_data['bias_sem'], plotting_data['bias_sem'][0])

                color = colors_list[list(sessions_dict.keys()).index(dataset_name)]
                
                # Plot with consistent styling across all plot types
                ax.plot(angles_rad, plot_data, marker='o', label=dataset_name, color=color)

                if error_bars == 'SEM':
                    # Consistent error bar styling
                    ax.fill_between(angles_rad, plot_data - error_data, plot_data + error_data,
                                  alpha=0.3, color=lighten_color(color))

            elif plot_mode == 'linear_comparison':
                # Rest of linear comparison code remains the same
                if plot_type == 'hit_rate':
                    plot_data = plotting_data['performance']
                    error_data = plotting_data['performance_sem']
                elif plot_type == 'bias_corrected':
                    plot_data = plotting_data['bias_corrected']
                    error_data = plotting_data['bias_corrected_sem']
                else:  # bias
                    plot_data = plotting_data['bias']
                    error_data = plotting_data['bias_sem']

                color = colors_list[list(sessions_dict.keys()).index(dataset_name)]
                ax.plot(angles_deg, plot_data, marker='o', label=dataset_name, color=color)

                if error_bars == 'SEM':
                    ax.fill_between(angles_deg, plot_data - error_data, plot_data + error_data,
                                  alpha=0.3, color=lighten_color(color))

    # Final plot adjustments
    ax.legend(loc='upper right')
    ax.set_title(plot_title)
    
    if plot_mode == 'radial':
        # Set y-axis limits based on plot type
        if plot_type == 'bias':
            ax.set_ylim(0, max(plot_data) * 1.1)  # Adjust for bias values
        elif plot_type == 'bias_corrected':
            ax.set_ylim(0, max(plot_data) * 1.1)  # Adjust for bias-corrected values
        else:
            ax.set_ylim(0, 1)  # Standard range for hit rate
            
        # Consistent radial plot styling for all plot types
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_xlim(np.radians(limits[0]), np.radians(limits[1]))
        
        # Set consistent angle labels for all plot types
        angles_label = np.arange(limits[0], limits[1] + bin_size, bin_size)
        ax.set_xticks(np.radians(angles_label))
        ax.set_xticklabels([f'{int(angle)}°' for angle in angles_label])
        
        # Ensure gridlines and axis ticks are consistent
        ax.grid(True)
        
    elif plot_mode == 'linear_comparison':
        ax.set_xlabel(x_title or 'Turn Angle (degrees)')
        ax.set_ylabel(y_title or ('Bias' if plot_type == 'bias' else 'Performance'))
        ax.set_xlim(limits[0], limits[1])
        if plot_type == 'bias_corrected':
            ax.set_ylim(0, max(plot_data) * 1.1)
        elif plot_type != 'bias':
            ax.set_ylim(0, 1)

    # Save plots if output path provided
    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cue_modes_str = '_'.join(cue_modes)
        
        if draft:
            base_filename = f"{date_time}_{plot_save_name}_{plot_type}_{cue_modes_str}"
        else:
            base_filename = f"final_{plot_save_name}_{plot_type}_{cue_modes_str}"
            
        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"

        # Handle existing files
        counter = 0
        while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
            output_filename_svg = f"{base_filename}_{counter}.svg"
            output_filename_png = f"{base_filename}_{counter}.png"
            counter += 1

        print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
        plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

        # Save the plot as PNG in the desired folder
        print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
        plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)
    
    # Display the plot
    plt.show()