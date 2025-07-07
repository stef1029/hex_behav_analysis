from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.cm as cm

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session

# Define your colours
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

# Function to lighten a colour for shaded regions
def lighten_colour(colour, factor=0.5):
    """
    Lighten a colour by blending it with white.
    
    Parameters:
    -----------
    colour : tuple
        RGB colour tuple with values between 0 and 1
    factor : float
        Lightening factor between 0 and 1
        
    Returns:
    --------
    tuple
        Lightened RGB colour tuple
    """
    return tuple(min(1, c + (1 - c) * factor) for c in colour)

def plot_performance_by_angle(sessions_input, 
                              plot_title='title',
                              x_title='',
                              y_title='', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              plot_type='hit_rate',  # 'hit_rate', 'bias_corrected', 'bias', or 'bias_incorrect'
                              cue_modes=['all_trials'],
                              error_bars='SEM',
                              plot_individual_mice=False,
                              exclusion_mice=[],
                              output_path=None,
                              plot_save_name='untitled_plot',
                              draft=True,
                              likelihood_threshold=0.6):
    """
    Plot angular performance data from behavioural sessions.
    
    This function takes a list of sessions or a dictionary of session lists 
    and plots angular performance data with various analysis options.

    Parameters:
    -----------
    sessions_input : list or dict
        List of sessions or dictionary of session lists
    plot_title : str
        Title for the plot
    x_title : str
        X-axis label
    y_title : str
        Y-axis label
    bin_mode : str
        Binning mode ('manual', 'rice', or 'tpb')
    num_bins : int
        Number of bins for manual mode
    trials_per_bin : int
        Target trials per bin for 'tpb' mode
    plot_mode : str
        Plot style ('radial' or 'linear_comparison')
    plot_type : str
        Type of plot to generate:
        - 'hit_rate': Raw performance/hit rate
        - 'bias_corrected': Bias-corrected performance
        - 'bias': Raw response bias (all trials included)
        - 'bias_incorrect': Bias histogram using only incorrect trials
    cue_modes : list
        List of cue modes to analyse
    error_bars : str
        Type of error bars ('SEM')
    plot_individual_mice : bool
        Whether to plot individual mouse data
    exclusion_mice : list
        List of mice to exclude from analysis
    output_path : Path
        Path to save output files
    plot_save_name : str
        Base name for saved files
    draft : bool
        Whether this is a draft plot
    likelihood_threshold : float
        Threshold for ear detection likelihood (default: 0.6)
    """

    def calc_performance(bins):
        """
        Convert a dict of lists of 0/1 correctness into mean performance per bin.
        
        Parameters:
        -----------
        bins : dict
            Dictionary mapping bin keys to lists of correctness values
            
        Returns:
        --------
        list
            Mean performance for each bin
        """
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]

    def calc_bias(bins, total_trials):
        """
        Calculate bias as proportion of touches to each angle.
        
        Parameters:
        -----------
        bins : dict
            Dictionary mapping angle bins to lists of touch counts
        total_trials : int
            Total number of trials for normalisation
            
        Returns:
        --------
        list
            Normalised bias values for each bin
        """
        bias_values = [sum(bins[key]) / total_trials if bins[key] else 0 for key in sorted(bins)]
        # Normalise so sum equals 1
        total = sum(bias_values)
        return [v / total if total > 0 else 0 for v in bias_values]

    def get_trials(sessions):
        """
        Collect all trial data, organised by mouse -> cue_mode -> trials.
        
        Parameters:
        -----------
        sessions : list
            List of session objects
            
        Returns:
        --------
        tuple
            (mice_dict, total_trials_dict) containing organised trial data
        """
        mice = {}
        total_trials = {mode: [] for mode in cue_modes}
        
        print(f"\n=== TRIAL COLLECTION DEBUG ===")
        print(f"Processing {len(sessions)} sessions")
        print(f"Cue modes: {cue_modes}")
        print(f"Exclusion mice: {exclusion_mice}")
        
        session_count = 0
        for session in sessions:
            session_count += 1
            mouse = session.session_dict.get('mouse_id', 'unknown')
            print(f"\nSession {session_count}: Mouse {mouse}")
            
            if mouse in exclusion_mice:
                print(f"  -> Excluding mouse {mouse}")
                continue
                
            if mouse not in mice:
                mice[mouse] = {mode: {'trials': [], 'bins': {}, 'bias_bins': {}, 'bias_incorrect_bins': {}} 
                               for mode in cue_modes}
                print(f"  -> Added new mouse {mouse}")
            
            trial_counts = {mode: 0 for mode in cue_modes}
            catch_trial_count = 0
            
            for trial in session.trials:
                # Exclude catch trials
                if trial.get('catch', False):
                    catch_trial_count += 1
                    continue

                # Distribute trials across the chosen cue_modes
                if "all_trials" in cue_modes:
                    mice[mouse]['all_trials']['trials'].append(trial)
                    total_trials['all_trials'].append(trial)
                    trial_counts['all_trials'] += 1
                
                if "visual_trials" in cue_modes and 'audio' not in trial.get('correct_port', ''):
                    mice[mouse]['visual_trials']['trials'].append(trial)
                    total_trials['visual_trials'].append(trial)
                    trial_counts['visual_trials'] += 1
                
                if "audio_trials" in cue_modes and 'audio' in trial.get('correct_port', ''):
                    mice[mouse]['audio_trials']['trials'].append(trial)
                    total_trials['audio_trials'].append(trial)
                    trial_counts['audio_trials'] += 1
            
            print(f"  -> Catch trials excluded: {catch_trial_count}")
            for mode in cue_modes:
                if mode in trial_counts:
                    print(f"  -> {mode}: {trial_counts[mode]} trials")
        
        print(f"\n=== FINAL TRIAL SUMMARY ===")
        print(f"Total mice included: {len(mice)}")
        for mouse in mice:
            print(f"\nMouse {mouse}:")
            for mode in cue_modes:
                print(f"  {mode}: {len(mice[mouse][mode]['trials'])} trials")
        
        print(f"\nAggregated totals:")
        for mode in cue_modes:
            print(f"  {mode}: {len(total_trials[mode])} trials across all mice")
        
        return mice, total_trials

    # Initialize plot
    if plot_mode == 'radial':
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    else:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

    # Handle colour selection for multiple datasets
    def get_colours(number_of_sessions):
        """
        Get colour palette for multiple datasets.
        
        Parameters:
        -----------
        number_of_sessions : int
            Number of datasets to colour
            
        Returns:
        --------
        list
            List of colour tuples
        """
        if number_of_sessions <= 3:
            return [colors['all_trials'], colors['visual_trials'], colors['audio_trials']][:number_of_sessions]
        else:
            cmap = plt.cm.get_cmap('viridis', number_of_sessions)
            return [cmap(i) for i in range(number_of_sessions)]

    # Decide how to interpret sessions_input
    if isinstance(sessions_input, dict):
        if cue_modes != ['all_trials']:
            raise ValueError("When providing a sessions dictionary, cue_modes must be ['all_trials'].")
        sessions_dict = sessions_input
        colours_list = get_colours(len(sessions_dict))
    else:
        sessions_dict = {'Data': sessions_input}
        colours_list = get_colours(1)

    print(f"\n=== DATASET PROCESSING ===")
    print(f"Number of datasets: {len(sessions_dict)}")
    print(f"Dataset names: {list(sessions_dict.keys())}")

    # Process each dataset
    for dataset_idx, (dataset_name, sessions) in enumerate(sessions_dict.items()):
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        data_sets = {}
        data_sets['mice'], data_sets['total_trials'] = get_trials(sessions)

        # Count total number of trials across relevant modes to pick bin size
        n = sum(len(data_sets['total_trials'][mode]) for mode in cue_modes)
        print(f"\nTotal trials for binning calculation: {n}")
        
        if bin_mode == 'manual':
            num_bins_used = num_bins
        elif bin_mode == 'rice':
            num_bins_used = int(2 * n ** (1/3))
        elif bin_mode == 'tpb':
            num_bins_used = int(n / trials_per_bin)
        else:
            raise ValueError('bin_mode must be "manual", "rice", or "tpb"')

        print(f"Binning mode: {bin_mode}")
        print(f"Number of bins used: {num_bins_used}")

        # Set angle limits based on plot_mode
        if plot_mode in ['linear_comparison', 'bar_split', 'bar_split_overlay']:
            limits = (0, 180)
            # Typically you'd want fewer bins if you're only covering 0-180,
            # but you can override as you like. For demonstration, let's do 6 bins:
            num_bins_used = 6
        else:
            limits = (-180, 180)

        angle_range = limits[1] - limits[0]
        bin_size = angle_range / num_bins_used

        print(f"Angle limits: {limits}")
        print(f"Bin size: {bin_size:.2f} degrees")

        # For each cue mode, collect performance/bias/bias_incorrect
        for cue_group in cue_modes:
            print(f"\n=== PROCESSING CUE GROUP: {cue_group} ===")
            
            # Prepare structure for storing average stats across mice
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
                'bias_incorrect': [],
                'bias_incorrect_sd': [],
                'bias_incorrect_sem': [],
                'n': [],
                'bin_titles': []
            }

            # ----- Populate bins for each mouse -----
            mouse_stats = {}
            for mouse in data_sets['mice']:
                print(f"\n--- Processing mouse: {mouse} ---")
                
                # Dictionary of angle bin -> list of correctness (1 or 0)
                perf_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                # For standard bias, we count *any* direction touches
                bias_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                # For "bias_incorrect", count only from incorrect trials
                bias_incorrect_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                # Track statistics for this mouse
                trials_processed = 0
                trials_skipped_no_turn_data = 0
                trials_skipped_low_likelihood = 0
                trials_with_performance_data = 0
                trials_with_bias_data = 0
                trials_incorrect_with_bias_data = 0

                # Go through each trial
                for trial in data_sets['mice'][mouse][cue_group]['trials']:
                    trials_processed += 1
                    
                    # If there's no "turn_data", skip
                    if trial.get("turn_data") is None:
                        trials_skipped_no_turn_data += 1
                        continue

                    # Check ear detection likelihood
                    left_likelihood = trial["turn_data"].get("left_ear_likelihood", 1)
                    right_likelihood = trial["turn_data"].get("right_ear_likelihood", 1)
                    
                    if (left_likelihood < likelihood_threshold or right_likelihood < likelihood_threshold):
                        trials_skipped_low_likelihood += 1
                        continue

                    angle = trial["turn_data"]["cue_presentation_angle"]
                    port_touched_angle = trial["turn_data"].get("port_touched_angle")

                    # Figure out if trial was correct or not
                    is_correct = False
                    if trial.get("next_sensor"):
                        is_correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])

                    # Fill in performance bins
                    for bin_start in perf_bins:
                        if bin_start <= angle < bin_start + bin_size:
                            perf_bins[bin_start].append(1 if is_correct else 0)
                            trials_with_performance_data += 1
                            break

                    # Fill in "standard" bias bins
                    if port_touched_angle is not None:
                        for bin_start in bias_bins:
                            if bin_start <= port_touched_angle < bin_start + bin_size:
                                bias_bins[bin_start].append(1)
                                trials_with_bias_data += 1
                                break

                    # Fill in "incorrect-only" bias bins
                    if (not is_correct) and (port_touched_angle is not None):
                        for bin_start in bias_incorrect_bins:
                            if bin_start <= port_touched_angle < bin_start + bin_size:
                                bias_incorrect_bins[bin_start].append(1)
                                trials_incorrect_with_bias_data += 1
                                break

                print(f"  Total trials for mouse: {trials_processed}")
                print(f"  Skipped (no turn data): {trials_skipped_no_turn_data}")
                print(f"  Skipped (low likelihood): {trials_skipped_low_likelihood}")
                print(f"  Used for performance: {trials_with_performance_data}")
                print(f"  Used for bias: {trials_with_bias_data}")
                print(f"  Used for incorrect bias: {trials_incorrect_with_bias_data}")

                # Show bin distributions
                print(f"  Performance bins (trials per bin):")
                for bin_start in sorted(perf_bins.keys()):
                    bin_centre = bin_start + bin_size/2
                    print(f"    {bin_centre:6.1f}°: {len(perf_bins[bin_start])} trials")

                # Store the aggregated bins back to the mouse's record if needed
                data_sets['mice'][mouse][cue_group].update({
                    'perf_bins': perf_bins,
                    'bias_bins': bias_bins,
                    'bias_incorrect_bins': bias_incorrect_bins
                })

                # Compute performance for each bin
                mouse_performance = calc_performance(perf_bins)

                # Standard bias: all trials
                total_trials_for_bias = len(data_sets['mice'][mouse][cue_group]['trials'])
                mouse_bias = calc_bias(bias_bins, total_trials_for_bias)

                # "Incorrect-only" bias:
                num_incorrect_trials = 0
                for tr in data_sets['mice'][mouse][cue_group]['trials']:
                    if tr.get("next_sensor"):
                        was_correct = (int(tr["correct_port"][-1]) == int(tr["next_sensor"]["sensor_touched"][-1]))
                        if not was_correct:
                            num_incorrect_trials += 1

                mouse_bias_incorrect = calc_bias(bias_incorrect_bins, num_incorrect_trials if num_incorrect_trials > 0 else 1)

                # Bias-corrected performance
                # (divide performance by the fraction of trials that touched that bin, with a small epsilon)
                mouse_bias_corrected = np.array(mouse_performance) / (np.array(mouse_bias) + 1e-10)

                # Store numeric arrays in the mouse record
                data_sets['mice'][mouse][cue_group].update({
                    'performance': mouse_performance,
                    'bias': mouse_bias,
                    'bias_incorrect': mouse_bias_incorrect,
                    'bias_corrected': mouse_bias_corrected.tolist()
                })

                # Store stats for summary
                mouse_stats[mouse] = {
                    'trials_processed': trials_processed,
                    'trials_used': trials_with_performance_data,
                    'mean_performance': np.mean(mouse_performance),
                    'mean_bias': np.mean(mouse_bias),
                    'num_incorrect_trials': num_incorrect_trials
                }

            # Print mouse summary
            print(f"\n=== MOUSE SUMMARY FOR {cue_group} ===")
            for mouse, stats in mouse_stats.items():
                print(f"Mouse {mouse}:")
                print(f"  Trials processed: {stats['trials_processed']}")
                print(f"  Trials used: {stats['trials_used']}")
                print(f"  Mean performance: {stats['mean_performance']:.3f}")
                print(f"  Mean bias: {stats['mean_bias']:.3f}")
                print(f"  Incorrect trials: {stats['num_incorrect_trials']}")

            # ----- Compute across-mice statistics -----
            print(f"\n=== COMPUTING ACROSS-MICE STATISTICS ===")
            
            # Convert each measure to an array [mouse x angle_bin]
            performance_array = np.array([
                data_sets['mice'][m][cue_group]['performance'] for m in data_sets['mice']
            ])
            bias_array = np.array([
                data_sets['mice'][m][cue_group]['bias'] for m in data_sets['mice']
            ])
            bias_incorrect_array = np.array([
                data_sets['mice'][m][cue_group]['bias_incorrect'] for m in data_sets['mice']
            ])
            bias_corrected_array = np.array([
                data_sets['mice'][m][cue_group]['bias_corrected'] for m in data_sets['mice']
            ])

            n_mice = len(data_sets['mice'])
            print(f"Number of mice: {n_mice}")
            print(f"Performance array shape: {performance_array.shape}")
            print(f"Bias array shape: {bias_array.shape}")

            # Means + spreads
            mean_performance = performance_array.mean(axis=0)
            sem_performance = performance_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias = bias_array.mean(axis=0)
            sem_bias = bias_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias_incorrect = bias_incorrect_array.mean(axis=0)
            sem_bias_incorrect = bias_incorrect_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias_corrected = bias_corrected_array.mean(axis=0)
            sem_bias_corrected = bias_corrected_array.std(axis=0) / np.sqrt(n_mice)

            print(f"Mean performance across bins: {mean_performance}")
            print(f"Mean bias across bins: {mean_bias}")

            # Fill in plotting_data with aggregated stats
            # Sorted by bin_start in ascending order
            bin_list = sorted(list(data_sets['mice'][mouse][cue_group]['perf_bins'].keys()))
            plotting_data.update({
                'performance': mean_performance,
                'performance_sd': performance_array.std(axis=0),
                'performance_sem': sem_performance,
                'bias': mean_bias,
                'bias_sd': bias_array.std(axis=0),
                'bias_sem': sem_bias,
                'bias_incorrect': mean_bias_incorrect,
                'bias_incorrect_sd': bias_incorrect_array.std(axis=0),
                'bias_incorrect_sem': sem_bias_incorrect,
                'bias_corrected': mean_bias_corrected,
                'bias_corrected_sd': bias_corrected_array.std(axis=0),
                'bias_corrected_sem': sem_bias_corrected,
                'n': n_mice,
                'bin_titles': [f"{b + (bin_size / 2):.2f}" for b in bin_list]
            })

            print(f"Bin centres: {plotting_data['bin_titles']}")

            # ----- Plotting -----
            print(f"\n=== PLOTTING ===")
            print(f"Plot type: {plot_type}")
            print(f"Plot mode: {plot_mode}")
            
            angles_deg = np.array(plotting_data['bin_titles'], dtype=np.float64)

            if plot_mode == 'radial':
                # Convert angles to [0..360), then wrap
                adjusted_angles_deg = angles_deg % 360
                angles_rad = np.radians(adjusted_angles_deg)
                # Append first value to the end to close the loop visually
                angles_rad = np.append(angles_rad, angles_rad[0])

                if plot_type == 'hit_rate':
                    plot_data = np.append(plotting_data['performance'], plotting_data['performance'][0])
                    error_data = np.append(plotting_data['performance_sem'], plotting_data['performance_sem'][0])

                elif plot_type == 'bias_corrected':
                    plot_data = np.append(plotting_data['bias_corrected'], plotting_data['bias_corrected'][0])
                    error_data = np.append(plotting_data['bias_corrected_sem'], plotting_data['bias_corrected_sem'][0])

                elif plot_type == 'bias_incorrect':
                    plot_data = np.append(plotting_data['bias_incorrect'], plotting_data['bias_incorrect'][0])
                    error_data = np.append(plotting_data['bias_incorrect_sem'], plotting_data['bias_incorrect_sem'][0])

                else:  # 'bias' by default
                    plot_data = np.append(plotting_data['bias'], plotting_data['bias'][0])
                    error_data = np.append(plotting_data['bias_sem'], plotting_data['bias_sem'][0])

                print(f"Plot data range: {np.min(plot_data):.3f} to {np.max(plot_data):.3f}")

                # Pick colour for the line
                colour = colors.get(cue_group, (0.5, 0.5, 0.5)) \
                    if len(cue_modes) > 1 else colours_list[list(sessions_dict.keys()).index(dataset_name)]

                # Decide label
                if len(cue_modes) > 1:
                    label = cue_group
                else:
                    label = cue_group

                # Plot
                ax.plot(angles_rad, plot_data, marker='o', label=label, color=colour)
                if error_bars == 'SEM':
                    ax.fill_between(angles_rad, plot_data - error_data, plot_data + error_data,
                                    alpha=0.3, color=lighten_colour(colour))

            elif plot_mode == 'linear_comparison':
                # Decide which metric to plot
                if plot_type == 'hit_rate':
                    plot_data = plotting_data['performance']
                    error_data = plotting_data['performance_sem']
                elif plot_type == 'bias_corrected':
                    plot_data = plotting_data['bias_corrected']
                    error_data = plotting_data['bias_corrected_sem']
                elif plot_type == 'bias_incorrect':
                    plot_data = plotting_data['bias_incorrect']
                    error_data = plotting_data['bias_incorrect_sem']
                else:  # 'bias'
                    plot_data = plotting_data['bias']
                    error_data = plotting_data['bias_sem']

                print(f"Plot data range: {np.min(plot_data):.3f} to {np.max(plot_data):.3f}")

                # Pick colour for the line
                colour = colors.get(cue_group, (0.5, 0.5, 0.5)) \
                    if len(cue_modes) > 1 else colours_list[list(sessions_dict.keys()).index(dataset_name)]

                # Decide label
                if len(cue_modes) > 1:
                    label = cue_group
                else:
                    label = f"{dataset_name} - {cue_group}"

                # Plot
                ax.plot(angles_deg, plot_data, marker='o', label=label, color=colour)
                if error_bars == 'SEM':
                    ax.fill_between(angles_deg, plot_data - error_data, plot_data + error_data,
                                    alpha=0.3, color=lighten_colour(colour))

    # Final plot cosmetics
    ax.legend(loc='upper right')
    ax.set_title(plot_title)
    
    if plot_mode == 'radial':
        # Set y-limits
        if plot_type == 'bias' or plot_type == 'bias_incorrect' or plot_type == 'bias_corrected':
            ax.set_ylim(0, max(plot_data) * 1.1 if len(plot_data) else 1)
        else:
            ax.set_ylim(0, 1)  # For standard hit_rate

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_xlim(np.radians(limits[0]), np.radians(limits[1]))

        # Create angle labels
        # angles_label = np.arange(limits[0], limits[1] + bin_size, bin_size)
        angles_label = np.arange(-180, 181, 30)
        # Avoid overlapping +180 / -180
        if len(angles_label) > 1 and angles_label[-1] == 180:
            angles_label = angles_label[:-1]

        ax.set_xticks(np.radians(angles_label))
        ax.set_xticklabels([f'{int(a)}°' for a in angles_label])
        ax.grid(True)

    elif plot_mode == 'linear_comparison':
        ax.set_xlabel(x_title or 'Turn Angle (degrees)')
        if plot_type in ('bias', 'bias_incorrect'):
            ax.set_ylabel(y_title or 'Bias')
        elif plot_type == 'bias_corrected':
            ax.set_ylabel(y_title or 'Bias-Corrected Performance')
        else:
            ax.set_ylabel(y_title or 'Performance')
        ax.set_xlim(limits[0], limits[1])

        # Rescale Y axis if needed
        if plot_type == 'bias_corrected':
            ax.set_ylim(0, max(plot_data) * 1.1 if len(plot_data) else 1)
        elif plot_type not in ('bias', 'bias_incorrect'):
            ax.set_ylim(0, 1)

    # Save if output_path is specified
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

        # If files already exist, append a counter
        counter = 0
        while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
            output_filename_svg = f"{base_filename}_{counter}.svg"
            output_filename_png = f"{base_filename}_{counter}.png"
            counter += 1

        print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
        plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

        print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
        plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)
    
    plt.show()
    print(f"\n=== PLOTTING COMPLETE ===")