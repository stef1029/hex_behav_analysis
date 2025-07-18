import numpy as np
from scipy import stats
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm

# Define your colours
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}


def calc_dprime(hits, total_signal_trials, false_alarms, total_noise_trials, adjustment=0.01):
    """
    Calculate d-prime (sensitivity index) based on signal detection theory.
    
    Parameters
    ----------
    hits : int
        Number of correct detections when signal was present
    total_signal_trials : int
        Total number of trials where signal was present
    false_alarms : int
        Number of incorrect detections when signal was absent
    total_noise_trials : int
        Total number of trials where signal was absent
    adjustment : float
        Small value to avoid division by zero and infinite values
        
    Returns
    -------
    float
        d-prime value, clipped between -5 and 5
    """
    if total_signal_trials == 0 or total_noise_trials == 0:
        return 0
        
    # Calculate rates with adjustment to avoid infinities
    hit_rate = (hits + adjustment) / (total_signal_trials + 2*adjustment)
    fa_rate = (false_alarms + adjustment) / (total_noise_trials + 2*adjustment)
    
    # Convert to d-prime
    try:
        dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        return np.clip(dprime, -5, 5)  # Limit extreme values
    except:
        return 0


def print_circular_statistics_summary(circular_stats_by_group, cue_modes):
    """
    Print a comprehensive summary of circular statistics for angular performance data.
    
    This function prints individual mouse statistics, group-level statistics, and 
    performs pairwise Watson-Williams tests between groups when multiple groups 
    are present.
    
    Parameters
    ----------
    circular_stats_by_group : dict
        Nested dictionary structure:
        {dataset_name: {cue_group: CircularStats object with attributes:
                                   - mouse_means: list
                                   - mouse_resultants: list
                                   - mouse_ids: list}}
    cue_modes : list
        List of cue mode strings (e.g., ['all_trials', 'visual_trials'])
        
    Returns
    -------
    dict
        Summary dictionary containing:
        - 'group_statistics': Group-level circular statistics
        - 'pairwise_comparisons': Results of all pairwise comparisons
        - 'significant_pairs': List of significantly different pairs
    """
    print(f"\n=== CIRCULAR STATISTICS (Performance-based) ===")
    
    # Store results for potential programmatic use
    summary_results = {
        'group_statistics': {},
        'pairwise_comparisons': [],
        'significant_pairs': []
    }
    
    # Process each cue mode
    for cue_group in cue_modes:
        if len(cue_modes) > 1:
            print(f"\nCue mode: {cue_group}")
        
        summary_results['group_statistics'][cue_group] = {}
        
        # Collect all groups for this cue mode
        groups_data = []
        group_names = []
        
        for dataset_name in circular_stats_by_group:
            if cue_group in circular_stats_by_group[dataset_name]:
                # Access dataclass attributes instead of dictionary keys
                stats = circular_stats_by_group[dataset_name][cue_group]
                if stats.mouse_means:  # Only include if has data
                    groups_data.append(stats.mouse_means)
                    group_names.append(dataset_name)
        
        # Debug output
        # print(f"\nFound {len(groups_data)} groups with data for {cue_group}")
        # if len(groups_data) < 2:
        #     print(f"  Note: Between-group comparison requires at least 2 groups (found {len(groups_data)})")
        # elif len(groups_data) > 2:
        #     n_comparisons = len(list(combinations(range(len(groups_data)), 2)))
        #     print(f"  Will perform {n_comparisons} pairwise comparisons")
        
        # Print individual mouse statistics
        for i, (group_name, group_means) in enumerate(zip(group_names, groups_data)):
            print(f"\n{group_name}:")
            # Access dataclass attributes
            stats = circular_stats_by_group[group_name][cue_group]
            
            # Store group results
            summary_results['group_statistics'][cue_group][group_name] = {
                'mouse_data': [],
                'group_mean': None,
                'group_concentration': None,
                'rayleigh_p': None
            }
            
            # Print individual mouse data
            # for mouse_id, mean, resultant in zip(stats.mouse_ids, 
            #                                      stats.mouse_means, 
            #                                      stats.mouse_resultants):
            #     print(f"  {mouse_id}: mean = {mean:.1f}°, R = {resultant:.3f}")
            #     summary_results['group_statistics'][cue_group][group_name]['mouse_data'].append({
            #         'mouse_id': mouse_id,
            #         'mean': mean,
            #         'resultant': resultant
            #     })
            
            # Calculate group-level statistics
            if stats.mouse_means:
                # Filter out NaN values
                valid_means = [m for m in stats.mouse_means if not np.isnan(m)]
                
                if valid_means:
                    group_circ_stats = calculate_circular_statistics(
                        valid_means, 
                        np.ones(len(valid_means))
                    )
                    print(f"  Group mean: {group_circ_stats['circular_mean']:.1f}°")
                    print(f"  Group concentration: R = {group_circ_stats['resultant_length']:.3f}")
                    print(f"  Rayleigh test p = {group_circ_stats['rayleigh_p']:.4f}")
                    
                    # Store group statistics
                    summary_results['group_statistics'][cue_group][group_name].update({
                        'group_mean': group_circ_stats['circular_mean'],
                        'group_concentration': group_circ_stats['resultant_length'],
                        'rayleigh_p': group_circ_stats['rayleigh_p']
                    })
                else:
                    print(f"  Group mean: No valid data")
                    print(f"  Group concentration: No valid data")
                    print(f"  Rayleigh test p = No valid data")
        
        # Perform between-group comparisons for all pairs if there are 2 or more groups
        if len(groups_data) >= 2:
            n_comparisons = len(list(combinations(range(len(groups_data)), 2)))
            print(f"\n=== BETWEEN-GROUP COMPARISONS ({n_comparisons} pairs) ===")
            
            # Generate all possible pairs
            for i, (group1_idx, group2_idx) in enumerate(combinations(range(len(groups_data)), 2)):
                group1_name = group_names[group1_idx]
                group2_name = group_names[group2_idx]
                group1_data = groups_data[group1_idx]
                group2_data = groups_data[group2_idx]
                
                print(f"\nComparison {i+1}: {group1_name} vs {group2_name}")
                ww_result = watson_williams_test(group1_data, group2_data)
                print(f"  {group1_name} mean: {ww_result['group1_mean']:.1f}°")
                print(f"  {group2_name} mean: {ww_result['group2_mean']:.1f}°")
                print(f"  F-statistic: {ww_result['F_statistic']:.3f}")
                print(f"  p-value: {ww_result['p_value']:.4f}")
                
                # Store comparison result
                comparison_result = {
                    'cue_group': cue_group,
                    'group1': group1_name,
                    'group2': group2_name,
                    'group1_mean': ww_result['group1_mean'],
                    'group2_mean': ww_result['group2_mean'],
                    'F_statistic': ww_result['F_statistic'],
                    'p_value': ww_result['p_value'],
                    'significant': ww_result['p_value'] < 0.05
                }
                summary_results['pairwise_comparisons'].append(comparison_result)
                
                if ww_result['p_value'] < 0.05:
                    print("  Result: Significant difference in circular means (p < 0.05)")
                else:
                    print("  Result: No significant difference in circular means (p >= 0.05)")
            
            # Summary of significant results
            print(f"\n=== SUMMARY ===")
            significant_pairs = []
            for i, (group1_idx, group2_idx) in enumerate(combinations(range(len(groups_data)), 2)):
                group1_data = groups_data[group1_idx]
                group2_data = groups_data[group2_idx]
                ww_result = watson_williams_test(group1_data, group2_data)
                if ww_result['p_value'] < 0.05:
                    pair_description = f"{group_names[group1_idx]} vs {group_names[group2_idx]} (p={ww_result['p_value']:.4f})"
                    significant_pairs.append(pair_description)
                    summary_results['significant_pairs'].append({
                        'cue_group': cue_group,
                        'pair': pair_description,
                        'p_value': ww_result['p_value']
                    })
            
            if significant_pairs:
                print(f"Significant differences found in {len(significant_pairs)} pairs:")
                for pair in significant_pairs:
                    print(f"  - {pair}")
            else:
                print("No significant differences found between any pairs")
                
        elif len(groups_data) == 1:
            print(f"\n=== BETWEEN-GROUP COMPARISON ===")
            print(f"Only one group found - no comparison possible")
    
    return summary_results


def calc_hit_rate(bins):
    """
    Convert a dict of lists of 0/1 correctness into mean hit_rate per bin.
    
    Parameters
    ----------
    bins : dict
        Dictionary mapping bin keys to lists of correctness values
        
    Returns
    -------
    list
        Mean performance for each bin
    """
    return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]


def calc_bias(bins, total_trials):
    """
    Calculate bias as proportion of touches to each angle.
    
    Parameters
    ----------
    bins : dict
        Dictionary mapping angle bins to lists of touch counts
    total_trials : int
        Total number of trials for normalisation
        
    Returns
    -------
    list
        Normalised bias values for each bin
    """
    bias_values = [sum(bins[key]) / total_trials if bins[key] else 0 for key in sorted(bins)]
    # Normalise so sum equals 1
    total = sum(bias_values)
    return [v / total if total > 0 else 0 for v in bias_values]


def get_colours(number_of_sessions):
    """
    Get colour palette for multiple datasets.
    
    Parameters
    ----------
    number_of_sessions : int
        Number of datasets to colour
        
    Returns
    -------
    list
        List of colour tuples
    """
    if number_of_sessions <= 3:
        return [colors['all_trials'], colors['visual_trials'], colors['audio_trials']][:number_of_sessions]
    else:
        cmap = plt.cm.get_cmap('viridis', number_of_sessions)
        return [cmap(i) for i in range(number_of_sessions)]


def lighten_colour(colour, factor=0.5):
    """
    Lighten a colour by blending it with white.
    
    Parameters
    ----------
    colour : tuple
        RGB colour tuple with values between 0 and 1
    factor : float
        Lightening factor between 0 and 1
        
    Returns
    -------
    tuple
        Lightened RGB colour tuple
    """
    return tuple(min(1, c + (1 - c) * factor) for c in colour)


def calculate_circular_statistics(angles, weights=None):
    """
    Calculate circular mean and resultant vector length.
    
    Parameters
    ----------
    angles : array-like
        Angles in degrees
    weights : array-like, optional
        Weights for each angle (e.g., performance values)
    
    Returns
    -------
    dict
        Contains circular_mean, resultant_length, and rayleigh_p
    """
    if len(angles) == 0:
        return {
            'circular_mean': np.nan,
            'resultant_length': np.nan,
            'rayleigh_p': np.nan
        }
    
    # Convert to radians
    angles_rad = np.deg2rad(angles)
    
    if weights is None:
        weights = np.ones_like(angles)
    
    # Calculate weighted circular mean
    sin_sum = np.sum(weights * np.sin(angles_rad))
    cos_sum = np.sum(weights * np.cos(angles_rad))
    
    # Circular mean
    circular_mean = np.rad2deg(np.arctan2(sin_sum, cos_sum))
    
    # Resultant vector length (measure of concentration)
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    R_bar = R / np.sum(weights)  # Mean resultant length
    
    # Rayleigh test for uniformity
    n = np.sum(weights)
    rayleigh_z = n * R_bar**2
    rayleigh_p = np.exp(-rayleigh_z) * (1 + (2*rayleigh_z - rayleigh_z**2) / (4*n))
    
    return {
        'circular_mean': circular_mean,
        'resultant_length': R_bar,
        'rayleigh_p': rayleigh_p
    }


def watson_williams_test(group1_means, group2_means):
    """
    Simplified Watson-Williams test for equality of circular means.
    
    Parameters
    ----------
    group1_means : array-like
        Circular means for each mouse in group 1 (in degrees)
    group2_means : array-like
        Circular means for each mouse in group 2 (in degrees)
    
    Returns
    -------
    dict
        Contains test statistic and p_value
    """
    # Remove NaN values
    group1_means = np.array(group1_means)
    group2_means = np.array(group2_means)
    group1_means = group1_means[~np.isnan(group1_means)]
    group2_means = group2_means[~np.isnan(group2_means)]
    
    if len(group1_means) == 0 or len(group2_means) == 0:
        return {
            'F_statistic': np.nan,
            'p_value': np.nan,
            'group1_mean': np.nan,
            'group2_mean': np.nan
        }
    
    # Convert to radians
    angles1_rad = np.deg2rad(group1_means)
    angles2_rad = np.deg2rad(group2_means)
    
    # Calculate group statistics
    n1 = len(group1_means)
    n2 = len(group2_means)
    
    # Calculate resultant vectors for each group
    R1 = np.sqrt(np.sum(np.cos(angles1_rad))**2 + np.sum(np.sin(angles1_rad))**2)
    R2 = np.sqrt(np.sum(np.cos(angles2_rad))**2 + np.sum(np.sin(angles2_rad))**2)
    R_total = np.sqrt(np.sum(np.cos(np.concatenate([angles1_rad, angles2_rad])))**2 + 
                      np.sum(np.sin(np.concatenate([angles1_rad, angles2_rad])))**2)
    
    # Calculate test statistic (simplified version)
    # For a full implementation, consider using pycircstat
    k = 2  # number of groups
    N = n1 + n2
    
    # This is a simplified F-statistic
    F = ((N - k) * (R1 + R2 - R_total)) / ((k - 1) * (N - R1 - R2))
    
    # Calculate p-value
    p_value = 1 - stats.f.cdf(F, k - 1, N - k)
    
    return {
        'F_statistic': F,
        'p_value': p_value,
        'group1_mean': np.rad2deg(np.arctan2(np.mean(np.sin(angles1_rad)), 
                                             np.mean(np.cos(angles1_rad)))),
        'group2_mean': np.rad2deg(np.arctan2(np.mean(np.sin(angles2_rad)), 
                                             np.mean(np.cos(angles2_rad))))
    }


def draw_polar_arrow(ax, angle_rad, length, color, linewidth=2, alpha=0.8, 
                     head_width=0.1, head_length=0.05):
    """
    Draw an arrow in polar coordinates with properly oriented head.
    
    Parameters
    ----------
    ax : matplotlib axis
        Polar axis to draw on
    angle_rad : float
        Angle in radians
    length : float
        Length of arrow
    color : tuple or str
        Arrow colour
    linewidth : float
        Width of arrow shaft
    alpha : float
        Transparency
    head_width : float
        Width of arrow head as fraction of length
    head_length : float
        Length of arrow head as fraction of total length
    """
    # Draw arrow shaft
    ax.plot([angle_rad, angle_rad], [0, length], 
            color=color, linewidth=linewidth, alpha=alpha)
    
    # Calculate arrow head vertices
    head_length_abs = length * head_length
    head_width_abs = length * head_width
    
    # Arrow tip
    tip_r = length
    tip_theta = angle_rad
    
    # Base of arrow head
    base_r = length - head_length_abs
    
    # Convert to angular width at the base radius
    if base_r > 0:
        angular_width = head_width_abs / base_r
    else:
        angular_width = 0.1
    
    # Create arrow head vertices
    vertices = [
        (tip_theta, tip_r),  # Tip
        (tip_theta - angular_width/2, base_r),  # Left base
        (tip_theta + angular_width/2, base_r),  # Right base
    ]
    
    # Create and add polygon
    arrow_head = Polygon(vertices, closed=True, 
                        facecolor=color, edgecolor=color, 
                        alpha=alpha, transform=ax.transData)
    ax.add_patch(arrow_head)

def save_plot_and_statistics(output_path, plot_save_name, draft, plot_type, cue_modes, 
                           num_bins_used, trials_per_bin, bin_mode, plot_title, plot_mode,
                           sessions_input, likelihood_threshold, timeout_handling, 
                           min_trial_duration, error_bars, plot_individual_mice,
                           exclusion_mice, total_excluded_info, stats_summary):
    """
    Save plot files and statistics to the specified output directory.
    
    Parameters
    ----------
    output_path : Path
        Directory to save files to
    plot_save_name : str
        Base name for saved files
    draft : bool
        Whether this is a draft (affects filename and overwrite behaviour)
    plot_type : str
        Type of plot being saved
    cue_modes : list
        List of cue modes analysed
    num_bins_used : int
        Number of bins used in analysis
    trials_per_bin : int
        Target trials per bin (for 'tpb' mode)
    bin_mode : str
        Binning mode used
    plot_title : str
        Title of the plot
    plot_mode : str
        Plot mode used
    sessions_input : list or dict
        Original sessions input
    likelihood_threshold : float
        Ear detection likelihood threshold used
    timeout_handling : str or None
        Timeout handling method used
    min_trial_duration : float or None
        Minimum trial duration filter used
    error_bars : str
        Error bar type used
    plot_individual_mice : bool
        Whether individual mice were plotted
    exclusion_mice : list
        List of excluded mice
    total_excluded_info : ExclusionInfo
        Exclusion statistics
    stats_summary : dict or None
        Circular statistics summary (None if not calculated)
        
    Returns
    -------
    tuple
        (svg_path, png_path, stats_path, csv_path) - paths to saved files
        csv_path will be None if no pairwise comparisons exist
    """
    import json
    import csv
    from datetime import datetime
    
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
    if draft:
        counter = 0
        while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
            output_filename_svg = f"{base_filename}_{counter}.svg"
            output_filename_png = f"{base_filename}_{counter}.png"
            counter += 1

    # Save plot files
    print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
    plt.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

    print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
    plt.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)

    # Save statistics if available
    stats_path = None
    csv_path = None
    
    if stats_summary is not None:        
        # Create stats filename
        stats_filename = f"{base_filename}_stats.json"
        
        # Check if file exists and increment counter if needed
        if draft:
            counter = 0
            while (output_path / stats_filename).exists():
                stats_filename = f"{base_filename}_{counter}_stats.json"
                counter += 1
        
        # Prepare comprehensive data for saving
        stats_to_save = {
            'metadata': {
                'plot_title': plot_title,
                'plot_type': plot_type,
                'plot_mode': plot_mode,
                'cue_modes': cue_modes,
                'bin_mode': bin_mode,
                'num_bins_used': num_bins_used,
                'trials_per_bin': trials_per_bin if bin_mode == 'tpb' else None,
                'date_generated': datetime.now().isoformat(),
                'number_of_groups': len(sessions_input) if isinstance(sessions_input, dict) else 1,
                'groups': list(sessions_input.keys()) if isinstance(sessions_input, dict) else ['Data'],
                'likelihood_threshold': likelihood_threshold,
                'timeout_handling': timeout_handling,
                'min_trial_duration': min_trial_duration,
                'error_bars': error_bars,
                'plot_individual_mice': plot_individual_mice
            },
            'exclusion_info': {
                'catch_trials_excluded': total_excluded_info.catch,
                'too_quick_trials_excluded': total_excluded_info.too_quick,
                'timeout_trials_excluded': total_excluded_info.timeout_excluded,
                'mice_excluded_no_audio': total_excluded_info.no_audio_trials,
                'manually_excluded_mice': exclusion_mice
            },
            'statistics': stats_summary
        }
        
        # Save to JSON file
        stats_path = output_path / stats_filename
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2, default=str)
        
        print(f"\nSaving statistics to: '{stats_path}'")
        
        # Also save a simplified CSV for pairwise comparisons (if they exist)
        if stats_summary.get('pairwise_comparisons'):
            csv_filename = f"{base_filename}_comparisons.csv"
            
            # Check if CSV exists
            if draft: 
                counter = 0
                while (output_path / csv_filename).exists():
                    csv_filename = f"{base_filename}_{counter}_comparisons.csv"
                    counter += 1
            
            csv_path = output_path / csv_filename
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['cue_group', 'group1', 'group2', 'group1_mean', 
                              'group2_mean', 'F_statistic', 'p_value', 'significant']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for comparison in stats_summary['pairwise_comparisons']:
                    # Round numerical values for cleaner CSV
                    row = comparison.copy()
                    row['group1_mean'] = round(row['group1_mean'], 2)
                    row['group2_mean'] = round(row['group2_mean'], 2)
                    row['F_statistic'] = round(row['F_statistic'], 3)
                    row['p_value'] = round(row['p_value'], 4)
                    writer.writerow(row)
            
            print(f"Saving comparison table to: '{csv_path}'")

    return (output_path / output_filename_svg, 
            output_path / output_filename_png, 
            stats_path, 
            csv_path)