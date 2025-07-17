import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
from scipy import stats
from pathlib import Path

def plot_turn_completion_distribution(sessions,
                                    cue_time='100ms',
                                    plot_title='Distribution of Turn Completion',
                                    x_label='% of Turn Completed at Cue Offset',
                                    y_label='Density',
                                    offset=0,
                                    cue_mode='both',
                                    trial_type='all',  # 'all', 'successful', 'unsuccessful'
                                    plot_individual_mice=False,
                                    exclusion_mice=[],
                                    likelihood_threshold=0.6,
                                    plot_type='histogram',  # 'kde' or 'histogram' (defaults to histogram)
                                    bins=50,
                                    xlim='auto',  # 'auto', None, or tuple (min, max)
                                    output_path=None,
                                    plot_save_name='turn_distribution',
                                    draft=True,
                                    max_delta_angle=90):
    """
    Creates a distribution plot of turn completion percentages for individual trials.
    
    Parameters:
    -----------
    sessions : list
        List of session objects
    cue_time : str
        Cue time string (e.g., '100ms', 'unlimited')
    plot_title : str
        Title for the plot
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    offset : float
        Time offset after cue end to measure (in seconds)
    cue_mode : str
        Which trials to include: 'both', 'visual', or 'audio'
    trial_type : str
        Which trials to plot: 'all', 'successful', or 'unsuccessful'
    plot_individual_mice : bool
        Whether to show individual mouse distributions
    exclusion_mice : list
        List of mouse IDs to exclude
    likelihood_threshold : float
        Minimum likelihood for ear detection
    plot_type : str
        'kde' for kernel density estimation or 'histogram' for line plot of histogram (default: 'histogram')
    bins : int
        Number of bins for histogram
    xlim : str, None, or tuple
        X-axis limits:
        - 'auto': automatically determine reasonable limits excluding extreme outliers
        - None: use full data range
        - tuple (min, max): manual limits
    output_path : Path
        Path to save output files
    plot_save_name : str
        Base name for saved files
    draft : bool
        Whether this is a draft plot
    max_delta_angle : float
        Maximum absolute angle change (in degrees) to include trials. This is the
        change in the angle from the mouse to the cue between start and offset times.
        Trials where the mouse has changed its angle to the cue by more than this
        value will be excluded. Default is 90 degrees.
    """
    
    def get_trials(session_list):
        """Organises trials by mouse and filters based on cue_mode."""
        trials = {}
        total_trials = []
        
        for session in session_list:
            mouse = session.session_dict['mouse_id']
            
            if mouse in exclusion_mice:
                continue
            
            if mouse not in trials:
                trials[mouse] = {'trials': []}
            
            for trial in session.trials:
                should_add = False
                if cue_mode == 'both':
                    should_add = True
                elif cue_mode == 'visual' and 'audio' not in trial['correct_port']:
                    should_add = True
                elif cue_mode == 'audio' and 'audio' in trial['correct_port']:
                    should_add = True
                
                if should_add:
                    trial['session_object'] = session
                    trials[mouse]['trials'].append(trial)
                    total_trials.append(trial)
                    
        return total_trials, trials
    
    def calculate_turn_percentage(trial, cue_time_str):
        """
        Calculate turn percentage for a single trial.
        
        This function calculates what percentage of the required turn toward the cue
        the mouse has completed by the cue offset time.
        
        Returns:
            tuple: (percentage_turn, delta_angle) or (None, None) if calculation fails
        """
        session = trial['session_object']
        
        # Skip trials without required data
        if len(trial.get('video_frames', [])) == 0:
            return None, None
            
        dlc_data = trial.get('DLC_data')
        if dlc_data is None or 'timestamps' not in dlc_data:
            return None, None
            
        timestamps = dlc_data['timestamps']
        
        # Calculate cue offset time
        if trial.get('cue_end') is not None:
            cue_offset_time = trial['cue_end'] + offset
        else:
            if cue_time_str == 'unlimited':
                return None, None
            else:
                cue_duration_ms = int(cue_time_str.replace('ms', ''))
                cue_duration_s = cue_duration_ms / 1000.0
                cue_start_time = trial.get('cue_start', trial.get('trial_start', 0))
                cue_offset_time = cue_start_time + cue_duration_s + offset
        
        if 'turn_data' not in trial or trial['turn_data'] is None:
            return None, None
        
        # Get the initial angle to the cue (from mouse perspective at cue start)
        cue_angle_at_start = trial['turn_data'].get('cue_presentation_angle')
        if cue_angle_at_start is None or np.isnan(cue_angle_at_start):
            return None, None
        
        # Find the frame index closest to cue offset time
        index = np.searchsorted(timestamps, cue_offset_time, side='left')
        if index >= len(timestamps):
            index = len(timestamps) - 1
        elif index > 0 and (index == len(timestamps) or 
                           abs(timestamps.iloc[index-1] - cue_offset_time) < 
                           abs(timestamps.iloc[index] - cue_offset_time)):
            index = index - 1
        
        if index < 0 or index >= len(dlc_data):
            return None, None
        
        # Get coordinates at cue offset
        cue_offset_coords = dlc_data.iloc[index]
        
        # Calculate mouse heading at cue offset using ear positions
        left_ear = (cue_offset_coords["left_ear"]["x"], cue_offset_coords["left_ear"]["y"])
        left_ear_likelihood = cue_offset_coords["left_ear"]["likelihood"]
        right_ear = (cue_offset_coords["right_ear"]["x"], cue_offset_coords["right_ear"]["y"])
        right_ear_likelihood = cue_offset_coords["right_ear"]["likelihood"]
        
        if left_ear_likelihood < likelihood_threshold or right_ear_likelihood < likelihood_threshold:
            return None, None
        
        # Calculate heading angle using the same method as find_angles
        # Vector from left ear to right ear
        ear_vector_x = right_ear[0] - left_ear[0]
        ear_vector_y = right_ear[1] - left_ear[1]
        
        # Check ear distance
        ear_distance = math.sqrt(ear_vector_x**2 + ear_vector_y**2)
        
        if ear_distance < 50:  # Ears too close, likely tracking error
            trial['outlier_debug'] = {
                'reason': 'ears_too_close',
                'ear_distance': ear_distance,
                'mouse': session.session_dict['mouse_id']
            }
            return None, None
        
        # The head direction is perpendicular to the ear vector
        head_vector_x = ear_vector_y
        head_vector_y = -ear_vector_x
        
        # Calculate angle using atan2 with -y for video coordinates
        theta_rad = math.atan2(-head_vector_y, head_vector_x)
        theta_deg = math.degrees(theta_rad) % 360
        
        # Calculate midpoint and apply offset
        midpoint = ((left_ear[0] + right_ear[0])/2, (left_ear[1] + right_ear[1])/2)
        theta_rad = math.radians(theta_deg)
        
        # Add offset for nose position
        eyes_offset = 40
        new_midpoint = (
            midpoint[0] + eyes_offset * math.cos(theta_rad),
            midpoint[1] - eyes_offset * math.sin(theta_rad)
        )
        
        # Calculate angle to the cue port at offset time
        port_coordinates = session.port_coordinates
        correct_port = trial["correct_port"]
        if correct_port == "audio-1":
            correct_port = "1"
        port_index = int(correct_port[-1]) - 1
        
        # Get the cue port coordinates
        port_x, port_y = port_coordinates[port_index]
        
        # Calculate vector from mouse to port
        vector_x = port_x - new_midpoint[0]
        vector_y = port_y - new_midpoint[1]
        
        # Calculate absolute angle to port
        port_angle_rad = math.atan2(-vector_y, vector_x)
        
        # Calculate relative angle (port angle - mouse heading)
        mouse_heading_rad = np.deg2rad(theta_deg)
        relative_angle_rad = port_angle_rad - mouse_heading_rad
        cue_angle_at_offset = math.degrees(relative_angle_rad) % 360
        
        # Normalise angles to [-180, 180] range
        if cue_angle_at_offset > 180:
            cue_angle_at_offset -= 360
        
        cue_angle_at_start_norm = cue_angle_at_start % 360
        if cue_angle_at_start_norm > 180:
            cue_angle_at_start_norm -= 360
        
        # Skip if initial angle is too small (mouse already facing the cue)
        if abs(cue_angle_at_start_norm) < 1.0:
            return None, None
        
        # Calculate how much the angle to the cue has changed
        # This is the amount the mouse has turned toward (or away from) the cue
        angle_change = cue_angle_at_start_norm - cue_angle_at_offset
        
        # Calculate percentage of required turn completed
        percentage_turn = (angle_change / cue_angle_at_start_norm) * 100
        
        # Handle angle wrapping for extreme cases
        if abs(percentage_turn) > 500:
            # Check if we have angle wrapping issue
            if cue_angle_at_start_norm * cue_angle_at_offset < 0 and abs(angle_change) > 180:
                # Recalculate using the shorter path
                if angle_change > 0:
                    angle_change = angle_change - 360
                else:
                    angle_change = angle_change + 360
                percentage_turn = (angle_change / cue_angle_at_start_norm) * 100
        
        if np.isnan(percentage_turn) or np.isinf(percentage_turn):
            return None, None
        
        # Store debug info for extreme values
        if abs(percentage_turn) > 500:
            trial['outlier_debug'] = {
                'percentage_turn': percentage_turn,
                'start_angle': cue_angle_at_start,
                'cue_angle': cue_angle_at_offset,
                'delta_angle': angle_change,
                'mouse': trial['session_object'].session_dict['mouse_id']
            }
        
        # Return percentage and the actual angle change (not delta between start and offset cue angles)
        return percentage_turn, angle_change
    
    # Collect all trials
    total_trials, trials_by_mouse = get_trials(sessions)
    
    # Calculate turn percentages for all trials
    all_percentages = []
    mouse_percentages = defaultdict(list)
    outlier_trials = []
    excluded_by_delta_angle = 0
    
    for mouse, mouse_data in trials_by_mouse.items():
        for trial in mouse_data['trials']:
            # Filter by trial type
            if trial_type != 'all':
                is_successful = False
                if trial.get("next_sensor") != {}:
                    is_successful = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                
                if trial_type == 'successful' and not is_successful:
                    continue
                elif trial_type == 'unsuccessful' and (is_successful or trial.get("next_sensor") == {}):
                    continue
            
            percentage, angle_change = calculate_turn_percentage(trial, cue_time)
            
            # Check if we should exclude based on angle change
            if percentage is not None and angle_change is not None:
                if abs(angle_change) > max_delta_angle:
                    excluded_by_delta_angle += 1
                    continue
                
                all_percentages.append(percentage)
                mouse_percentages[mouse].append(percentage)
                
                # Collect outlier information
                if 'outlier_debug' in trial:
                    outlier_trials.append(trial['outlier_debug'])
    
    # Print outlier information
    if outlier_trials:
        print(f"\n=== OUTLIER TRIALS DETECTED ({len(outlier_trials)} trials) ===")
        
        # Separate by type
        small_angle_outliers = [o for o in outlier_trials if o.get('reason') == 'very_small_start_angle']
        extreme_value_outliers = [o for o in outlier_trials if 'percentage_turn' in o]
        
        if small_angle_outliers:
            print(f"\nTrials excluded due to very small start angles (<1°): {len(small_angle_outliers)}")
            for outlier in small_angle_outliers[:3]:
                print(f"  Mouse {outlier['mouse']}: start angle = {outlier['start_angle']:.3f}°")
        
        if extreme_value_outliers:
            print(f"\nExtreme percentage values: {len(extreme_value_outliers)} trials")
            # Sort by absolute percentage value
            extreme_value_outliers.sort(key=lambda x: abs(x['percentage_turn']), reverse=True)
            # Print top 10 most extreme
            for i, outlier in enumerate(extreme_value_outliers[:10]):
                print(f"\nOutlier {i+1}:")
                print(f"  Mouse: {outlier['mouse']}")
                print(f"  Percentage turn: {outlier['percentage_turn']:.1f}%")
                print(f"  Initial angle to cue: {outlier['start_angle']:.1f}°")
                print(f"  Final angle to cue: {outlier['cue_angle']:.1f}°")
                print(f"  Angle change: {outlier['delta_angle']:.1f}°")
            if len(extreme_value_outliers) > 10:
                print(f"\n  ... and {len(extreme_value_outliers) - 10} more extreme values")
    
    # Print exclusion information
    if excluded_by_delta_angle > 0:
        print(f"\nTrials excluded due to absolute angle change > {max_delta_angle}°: {excluded_by_delta_angle}")
    
    # Check if we have any data
    if len(all_percentages) == 0:
        print("No valid data points to plot!")
        return all_percentages, mouse_percentages
    
    # Remove extreme outliers for plotting (optional)
    percentiles = np.percentile(all_percentages, [1, 99])
    reasonable_range = (percentiles[0] - 1.5 * (percentiles[1] - percentiles[0]),
                       percentiles[1] + 1.5 * (percentiles[1] - percentiles[0]))
    
    # Filter data for plotting if specified
    plot_percentages = all_percentages
    filtered_count = 0
    
    if xlim == 'auto':
        # Auto-determine reasonable x-limits based on data
        plot_percentages = [p for p in all_percentages if reasonable_range[0] <= p <= reasonable_range[1]]
        filtered_count = len(all_percentages) - len(plot_percentages)
        xlim = (max(-100, np.percentile(plot_percentages, 1)), 
                min(200, np.percentile(plot_percentages, 99)))
    elif xlim is None:
        xlim = (min(all_percentages), max(all_percentages))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colours
    main_colour = (0, 0.68, 0.94)  # Blue
    mouse_colours = plt.cm.Set2(np.linspace(0, 1, len(mouse_percentages)))
    
    # Always calculate bin edges for consistency
    _, bin_edges = np.histogram(plot_percentages, bins=bins, range=xlim)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    x_range = np.linspace(xlim[0], xlim[1], 200)  # For KDE plotting
    
    # Plot based on plot_type
    if plot_type == 'kde' and len(plot_percentages) > 1:
        try:
            kde = stats.gaussian_kde(plot_percentages)
            x_range = np.linspace(xlim[0], xlim[1], 200)
            density = kde(x_range)
            ax.plot(x_range, density, color=main_colour, linewidth=2, 
                   label=f'All mice (n={len(all_percentages)} trials)')
        except:
            print("KDE failed, falling back to histogram")
            plot_type = 'histogram'
    
    if plot_type == 'histogram' or len(plot_percentages) <= 1:
        # Use histogram-based line plot
        counts, _ = np.histogram(plot_percentages, bins=bin_edges)
        
        # Normalise to density
        density = counts / (len(plot_percentages) * bin_width)
        
        # Plot main line
        ax.plot(bin_centers, density, color=main_colour, linewidth=2, 
               marker='o', markersize=4, label=f'All mice (n={len(all_percentages)} trials)')
    
    # Plot individual mice if requested
    if plot_individual_mice:
        for i, (mouse, percentages) in enumerate(sorted(mouse_percentages.items())):
            # Use the same filtered data approach for consistency
            if filtered_count > 0:
                mouse_plot_data = [p for p in percentages if reasonable_range[0] <= p <= reasonable_range[1]]
            else:
                mouse_plot_data = percentages
                
            if len(mouse_plot_data) > 0:
                if plot_type == 'kde' and len(mouse_plot_data) > 1:
                    try:
                        kde = stats.gaussian_kde(mouse_plot_data)
                        density = kde(x_range)
                        ax.plot(x_range, density, color=mouse_colours[i], linewidth=1.5, 
                               alpha=0.7, linestyle='--', label=f'{mouse} (n={len(percentages)})')
                    except:
                        # Fall back to histogram for this mouse
                        counts, _ = np.histogram(mouse_plot_data, bins=bin_edges)
                        density = counts / (len(mouse_plot_data) * bin_width)
                        ax.plot(bin_centers, density, color=mouse_colours[i], linewidth=1.5, 
                               alpha=0.7, linestyle='--', marker='o', markersize=3,
                               label=f'{mouse} (n={len(percentages)})')
                else:
                    counts, _ = np.histogram(mouse_plot_data, bins=bin_edges)
                    density = counts / (len(mouse_plot_data) * bin_width)
                    ax.plot(bin_centers, density, color=mouse_colours[i], linewidth=1.5, 
                           alpha=0.7, linestyle='--', marker='o', markersize=3,
                           label=f'{mouse} (n={len(percentages)})')
    
    # Add vertical lines for reference
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='No turn')
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='Complete turn')
    
    # Customise plot
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if plot_individual_mice:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax.legend(loc='best')
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(all_percentages):.1f}%\n'
    stats_text += f'Median: {np.median(all_percentages):.1f}%\n'
    stats_text += f'SD: {np.std(all_percentages):.1f}%'
    if filtered_count > 0:
        stats_text += f'\n{filtered_count} outliers hidden'
    if excluded_by_delta_angle > 0:
        stats_text += f'\n{excluded_by_delta_angle} trials excluded (|angle change| > {max_delta_angle}°)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if draft:
            base_filename = f"{date_time}_{plot_save_name}_{cue_time}_{trial_type}_{cue_mode}"
        else:
            base_filename = f"final_{plot_save_name}_{cue_time}_{trial_type}_{cue_mode}"
        
        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"
        
        # Check for existing files
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
    
    # Print summary statistics
    print(f"\n=== TURN COMPLETION DISTRIBUTION ===")
    print(f"Cue time: {cue_time}")
    print(f"Trial type: {trial_type}")
    print(f"Cue mode: {cue_mode}")
    print(f"Max delta angle filter: {max_delta_angle}°")
    print(f"Total trials analysed: {len(all_percentages)}")
    print(f"Mean: {np.mean(all_percentages):.1f}%")
    print(f"Median: {np.median(all_percentages):.1f}%")
    print(f"SD: {np.std(all_percentages):.1f}%")
    print(f"Range: {np.min(all_percentages):.1f}% to {np.max(all_percentages):.1f}%")
    
    if plot_individual_mice:
        print(f"\nIndividual mice:")
        for mouse in sorted(mouse_percentages.keys()):
            percentages = mouse_percentages[mouse]
            print(f"  {mouse}: n={len(percentages)}, mean={np.mean(percentages):.1f}%, SD={np.std(percentages):.1f}%")
    
    return all_percentages, mouse_percentages