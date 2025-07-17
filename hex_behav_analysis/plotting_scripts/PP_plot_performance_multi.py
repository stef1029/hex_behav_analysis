from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.cm as cm
from scipy import stats
from itertools import combinations
from matplotlib.patches import Polygon
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session

# Define your colours
colors = {
    "all_trials": (0, 0.68, 0.94),
    "visual_trials": (0.93, 0, 0.55),
    "audio_trials": (1, 0.59, 0)
}

from hex_behav_analysis.plotting_scripts.PP_plot_performance_multi_utils import *


@dataclass
class BinData:
    """Store binned data for hit rate, bias, and signal detection calculations."""
    hit_rate_bins: Dict[float, List[int]] = field(default_factory=dict)
    bias_bins: Dict[float, List[int]] = field(default_factory=dict)
    bias_incorrect_bins: Dict[float, List[int]] = field(default_factory=dict)
    signal_detection_bins: Dict[float, Dict[str, int]] = field(default_factory=dict)


@dataclass
class MouseCueModeData:
    """Store all data for a single mouse and cue mode combination."""
    trials: List[Dict[str, Any]] = field(default_factory=list)
    bins: BinData = field(default_factory=BinData)
    hit_rate: List[float] = field(default_factory=list)
    bias: List[float] = field(default_factory=list)
    bias_incorrect: List[float] = field(default_factory=list)
    bias_corrected: List[float] = field(default_factory=list)
    dprime: List[float] = field(default_factory=list)


@dataclass
class CircularStats:
    """Store circular statistics for a group."""
    mouse_means: List[float] = field(default_factory=list)
    mouse_resultants: List[float] = field(default_factory=list)
    mouse_ids: List[str] = field(default_factory=list)


@dataclass
class PlottingData:
    """Store aggregated statistics for plotting."""
    hit_rate: List[float] = field(default_factory=list)
    hit_rate_sd: List[float] = field(default_factory=list)
    hit_rate_sem: List[float] = field(default_factory=list)

    bias: List[float] = field(default_factory=list)
    bias_sd: List[float] = field(default_factory=list)
    bias_sem: List[float] = field(default_factory=list)

    bias_corrected: List[float] = field(default_factory=list)
    bias_corrected_sd: List[float] = field(default_factory=list)
    bias_corrected_sem: List[float] = field(default_factory=list)

    bias_incorrect: List[float] = field(default_factory=list)
    bias_incorrect_sd: List[float] = field(default_factory=list)
    bias_incorrect_sem: List[float] = field(default_factory=list)

    dprime: List[float] = field(default_factory=list)
    dprime_sd: List[float] = field(default_factory=list)
    dprime_sem: List[float] = field(default_factory=list)

    n: int = 0
    bin_titles: List[str] = field(default_factory=list)



@dataclass
class ExclusionInfo:
    """
    Track all exclusion information for trials and mice.
    """
    # Trial exclusion counts
    catch: int = 0
    too_quick: int = 0
    timeout_excluded: int = 0
    no_audio_trials: List[str] = field(default_factory=list)


@dataclass
class DataSet:
    """Store organised trial data for a dataset."""
    mice: Dict[str, Dict[str, MouseCueModeData]] = field(default_factory=dict)
    total_trials: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def plot_performance_by_angle(sessions_input, 
                              plot_title='title',
                              x_title='',
                              y_title='', 
                              bin_mode='manual', 
                              num_bins=12, 
                              trials_per_bin=10, 
                              plot_mode='radial', 
                              plot_type='hit_rate',  # 'hit_rate', 'bias_corrected', 'bias', 'bias_incorrect', 'dprime'
                              cue_modes=['all_trials'],
                              error_bars='SEM',
                              plot_individual_mice=False,
                              exclusion_mice=[],
                              output_path=None,
                              plot_save_name='untitled_plot',
                              draft=True,
                              likelihood_threshold=0.6,
                              timeout_handling=None,
                              min_trial_duration=None,
                              show_circular_stats=True,
                              plot_circular_means=True):
    """
    Plot angular performance data from behavioural sessions.
    
    This function takes a list of sessions or a dictionary of session lists 
    and plots angular performance data with various analysis options. When 
    multiple groups are provided (via dictionary input), it performs pairwise 
    circular statistics comparisons between all groups.

    Parameters
    ----------
    sessions_input : list or dict
        List of sessions or dictionary of session lists. If a dictionary is 
        provided with multiple groups (e.g., {'Control': [...], 'Test1': [...], 
        'Test2': [...]}), pairwise comparisons will be performed between all groups.
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
        - 'dprime': Signal detection sensitivity (d-prime)
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
    timeout_handling : str or None
        How to handle timeout trials:
        - None: Include timeouts as incorrect trials (default behaviour)
        - 'exclude': Exclude timeouts from hit rate calculations but keep in total trial count
        - 'exclude_total': Exclude timeouts from all calculations entirely
    min_trial_duration : float or None
        Minimum trial duration in seconds. Trials shorter than this will be excluded.
        If None, no duration-based filtering is applied.
    show_circular_stats : bool
        Whether to calculate and display circular statistics
    plot_circular_means : bool
        Whether to plot circular mean vectors on radial plots
    """

    def get_trials(sessions):
        """
        Collect all trial data, organised by mouse -> cue_mode -> trials.
        """
        mice = {}
        total_trials = {mode: [] for mode in cue_modes}
        exclusion_info = ExclusionInfo()
        
        for session in sessions:
            mouse = session.session_dict.get('mouse_id', 'unknown')
            if mouse in exclusion_mice:
                continue
            
            if mouse not in mice:
                mice[mouse] = {}
                for mode in cue_modes:
                    mice[mouse][mode] = MouseCueModeData()
            
            for trial in session.trials:
                # Exclude catch trials
                if trial.get('catch', False):
                    exclusion_info.catch += 1
                    continue
                
                # Calculate trial duration and exclude if too quick
                if min_trial_duration is not None:
                    cue_start = trial.get('cue_start')
                    next_sensor = trial.get('next_sensor', {})
                    
                    # Calculate trial duration if both timestamps exist
                    if cue_start is not None and next_sensor.get('sensor_start') is not None:
                        trial_duration = next_sensor['sensor_start'] - cue_start
                        if trial_duration < min_trial_duration:
                            exclusion_info.too_quick += 1
                            continue
                    # If there's no sensor touch (timeout), we can't calculate duration properly
                    # These will be handled by timeout_handling parameter
                
                # Handle timeout exclusion if specified
                if timeout_handling == 'exclude_total' and trial.get('next_sensor', {}) == {}:
                    exclusion_info.timeout_excluded += 1
                    continue

                # Distribute trials across the chosen cue_modes
                if "all_trials" in cue_modes:
                    mice[mouse]['all_trials'].trials.append(trial)
                    total_trials['all_trials'].append(trial)
                
                if "visual_trials" in cue_modes and 'audio' not in trial.get('correct_port', ''):
                    mice[mouse]['visual_trials'].trials.append(trial)
                    total_trials['visual_trials'].append(trial)
                
                if "audio_trials" in cue_modes and 'audio' in trial.get('correct_port', ''):
                    mice[mouse]['audio_trials'].trials.append(trial)
                    total_trials['audio_trials'].append(trial)
        
        # Post-processing: exclude mice with no trials for specific trial types
        mice_to_remove = []
        for key in mice:
            for cue_mode in cue_modes:
                if cue_mode == 'audio_trials' and len(mice[key][cue_mode].trials) == 0:
                    mice_to_remove.append(key)
                    exclusion_info.no_audio_trials.append(key)
                    break
        
        # Remove mice with insufficient trial types
        for key in mice_to_remove:
            del mice[key]
        
        return mice, total_trials, exclusion_info

    """
    Plotting logic flow starts here:
    """
    # ------------------------------------------------------------------
    # Initialise plot
    if plot_mode == 'radial':
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'polar': True})
    else:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
    # ------------------------------------------------------------------

    # Dictionary to hold mouse-specific data for individual plotting
    if plot_individual_mice:
        mouse_data_dict = {cue_group: {} for cue_group in cue_modes}

    # Dictionary to store circular statistics for each group
    circular_stats_by_group: Dict[str, Dict[str, CircularStats]] = {}

    # Decide how to interpret sessions_input - check to make sure that you're not trying to do multiple cue mode lines if you've got multiple datasets.
    if isinstance(sessions_input, dict):
        # Changed: Allow any single cue mode, not just 'all_trials'
        if len(cue_modes) != 1:
            raise ValueError("When providing a sessions dictionary, cue_modes must contain exactly one trial type.")
        sessions_dict = sessions_input
        colours_list = get_colours(len(sessions_dict))
    else:
        sessions_dict = {'Data': sessions_input}        # Converts a list of sessions into a single group
        colours_list = get_colours(1)
    # ------------------------------------------------------------------
    # Track total excluded trials across all datasets
    total_excluded_info = ExclusionInfo()
    # ------------------------------------------------------------------
    # Process each dataset (eg {'Control': [sess], 'Test1': [sess], 'Test2': [sess]})
    for dataset_name, sessions in sessions_dict.items():
        data_sets = DataSet()

        # Sort trials into mice -> cue_mode -> trials
        data_sets.mice, data_sets.total_trials, excluded_info = get_trials(sessions)
        
        # Accumulate exclusion counts
        total_excluded_info.catch += excluded_info.catch
        total_excluded_info.too_quick += excluded_info.too_quick
        total_excluded_info.timeout_excluded += excluded_info.timeout_excluded
        total_excluded_info.no_audio_trials.extend(excluded_info.no_audio_trials)

        # ---------------------- Set up binning 
        # Count total number of trials across relevant modes to pick bin size
        n = sum(len(data_sets.total_trials[mode]) for mode in cue_modes)
        if bin_mode == 'manual':
            num_bins_used = num_bins
        elif bin_mode == 'rice':
            num_bins_used = int(2 * n ** (1/3))
        elif bin_mode == 'tpb':
            num_bins_used = int(n / trials_per_bin)
        else:
            raise ValueError('bin_mode must be "manual", "rice", or "tpb"')

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
        # ----------------------

        # For each cue mode (visual trials, audio trials, all trials), collect hit_rate/bias/bias_incorrect/dprime
        for cue_group in cue_modes:
            # Prepare structure for storing average stats across mice
            plotting_data = PlottingData()

            # Store circular statistics for this group
            if dataset_name not in circular_stats_by_group:
                circular_stats_by_group[dataset_name] = {}
            circular_stats_by_group[dataset_name][cue_group] = CircularStats()

            # Create bins for each mouse
            for mouse_id in data_sets.mice:
                # Dictionary of angle bin -> list of correctness (1 or 0)
                hit_rate_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                # For standard bias, we count *any* direction touches
                bias_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}

                # For "bias_incorrect", count only from incorrect trials
                bias_incorrect_bins = {i: [] for i in np.arange(limits[0], limits[1], bin_size)}
                
                # For d-prime, track signal detection metrics
                signal_detection_bins = {i: {
                    'hits': 0,
                    'total_signal_trials': 0,
                    'false_alarms': 0,
                    'total_noise_trials': 0
                } for i in np.arange(limits[0], limits[1], bin_size)}

                # Track number of incorrect trials as we go
                num_incorrect_trials = 0

                # Go through each trial (SINGLE PASS)
                for trial in data_sets.mice[mouse_id][cue_group].trials:
                    # If there's no "turn_data", skip
                    if trial.get("turn_data") is None:
                        continue

                    # Check ear detection likelihood
                    if (trial["turn_data"].get("left_ear_likelihood", 1) < likelihood_threshold or
                        trial["turn_data"].get("right_ear_likelihood", 1) < likelihood_threshold):
                        continue

                    cue_presentation_angle = trial["turn_data"]["cue_presentation_angle"]
                    port_touched_angle = trial["turn_data"].get("port_touched_angle")

                    # Figure out if trial was correct or not
                    is_correct = False
                    is_timeout = trial.get("next_sensor") == {}
                    
                    if trial.get("next_sensor") and not is_timeout:
                        is_correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                        
                        # Count incorrect trials (considering timeout handling)
                        if not is_correct:
                            num_incorrect_trials += 1

                        # --------- D prime handling ---------
                        # Find cue bin and response bin
                        cue_bin = None
                        response_bin = None
                        
                        for bin_start in signal_detection_bins:
                            if bin_start <= cue_presentation_angle < bin_start + bin_size:
                                cue_bin = bin_start
                            if port_touched_angle is not None and bin_start <= port_touched_angle < bin_start + bin_size:
                                response_bin = bin_start
                        
                        if cue_bin is not None:
                            # This angle bin had a signal (cue presented here)
                            signal_detection_bins[cue_bin]['total_signal_trials'] += 1
                            
                            # Check if it was a hit (correct response to this angle)
                            if is_correct and cue_bin == response_bin:
                                signal_detection_bins[cue_bin]['hits'] += 1
                            
                            # For all other bins, this was a noise trial
                            for bin_start in signal_detection_bins:
                                if bin_start != cue_bin:
                                    signal_detection_bins[bin_start]['total_noise_trials'] += 1
                                    
                                    # Check for false alarms
                                    if bin_start == response_bin and not is_correct:
                                        signal_detection_bins[bin_start]['false_alarms'] += 1
                        # --------------------------------------
                    elif is_timeout and not (timeout_handling == 'exclude'):
                        # Timeout trials count as incorrect (unless excluded)
                        num_incorrect_trials += 1
                        
                    # Handle timeout trials based on timeout_handling parameter
                    if is_timeout and timeout_handling == 'exclude':
                        # Skip timeouts for hit rate calculations
                        pass
                    else:
                        # Fill in hit rate bins for non-timeout trials (or all trials if timeout_handling is None)
                        for bin_start in hit_rate_bins:
                            if bin_start <= cue_presentation_angle < bin_start + bin_size:
                                hit_rate_bins[bin_start].append(1 if is_correct else 0)
                                break

                    # Fill in "standard" bias bins
                    if port_touched_angle is not None:
                        for bin_start in bias_bins:
                            if bin_start <= port_touched_angle < bin_start + bin_size:
                                bias_bins[bin_start].append(1)
                                break

                    # Fill in "incorrect-only" bias bins
                    if is_timeout and timeout_handling == 'exclude':
                        # Skip timeouts when building incorrect-only bias
                        pass
                    elif (not is_correct) and (port_touched_angle is not None):
                        for bin_start in bias_incorrect_bins:
                            if bin_start <= port_touched_angle < bin_start + bin_size:
                                bias_incorrect_bins[bin_start].append(1)
                                break

                # Now calculate all metrics
                bin_list = sorted(list(hit_rate_bins.keys()))
                bin_centers = [b + bin_size / 2 for b in bin_list]

                # Calculate hit rate
                mouse_hit_rate = calc_hit_rate(hit_rate_bins)

                # Standard bias: all trials
                total_trials_for_bias = len(data_sets.mice[mouse_id][cue_group].trials)
                mouse_bias = calc_bias(bias_bins, total_trials_for_bias)

                # "Incorrect-only" bias using the count we already have
                mouse_bias_incorrect = calc_bias(bias_incorrect_bins, num_incorrect_trials if num_incorrect_trials > 0 else 1)

                # Bias-corrected hit rate
                mouse_bias_corrected = np.array(mouse_hit_rate) / (np.array(mouse_bias) + 1e-10)

                # Calculate d-prime for each bin
                mouse_dprime = []
                for bin_start in sorted(signal_detection_bins):
                    bin_data = signal_detection_bins[bin_start]
                    dprime = calc_dprime(
                        bin_data['hits'],
                        bin_data['total_signal_trials'],
                        bin_data['false_alarms'],
                        bin_data['total_noise_trials']
                    )
                    mouse_dprime.append(dprime)

                # NOW calculate circular statistics after all metrics are computed
                # Select the appropriate data based on plot type
                if plot_type == 'hit_rate':
                    data_for_stats = mouse_hit_rate
                elif plot_type == 'bias':
                    data_for_stats = mouse_bias
                elif plot_type == 'bias_corrected':
                    data_for_stats = mouse_bias_corrected.tolist()  # Convert numpy array to list
                elif plot_type == 'bias_incorrect':
                    data_for_stats = mouse_bias_incorrect
                elif plot_type == 'dprime':
                    data_for_stats = mouse_dprime
                else:
                    raise ValueError(f"Unknown plot_type: {plot_type}")

                # Filter out zero or negative values for weights (depending on plot type)
                valid_angles = []
                valid_weights = []

                for angle, value in zip(bin_centers, data_for_stats):
                    # Different filtering logic based on plot type
                    if plot_type == 'dprime':
                        # For d-prime, we might want to include negative values
                        # but shift them to ensure positive weights
                        if not np.isnan(value):
                            valid_angles.append(angle)
                            # Shift d-prime values to be positive (adding 3 ensures most values are positive)
                            valid_weights.append(value + 3)
                    elif plot_type in ['bias', 'bias_incorrect']:
                        # For bias metrics, only include non-zero values
                        if value > 0:
                            valid_angles.append(angle)
                            valid_weights.append(value)
                    else:
                        # For hit_rate and bias_corrected, include all positive values
                        if value > 0:
                            valid_angles.append(angle)
                            valid_weights.append(value)

                # Calculate circular statistics if we have valid data
                if valid_angles:
                    mouse_circ_stats = calculate_circular_statistics(valid_angles, valid_weights)
                    circular_stats_by_group[dataset_name][cue_group].mouse_means.append(
                        mouse_circ_stats['circular_mean'])
                    circular_stats_by_group[dataset_name][cue_group].mouse_resultants.append(
                        mouse_circ_stats['resultant_length'])
                    circular_stats_by_group[dataset_name][cue_group].mouse_ids.append(mouse_id)
                else:
                    # No valid data for this mouse - store NaN
                    circular_stats_by_group[dataset_name][cue_group].mouse_means.append(np.nan)
                    circular_stats_by_group[dataset_name][cue_group].mouse_resultants.append(np.nan)
                    circular_stats_by_group[dataset_name][cue_group].mouse_ids.append(mouse_id)

                # Store the bins and numeric arrays in the mouse record
                data_sets.mice[mouse_id][cue_group].bins.hit_rate_bins = hit_rate_bins
                data_sets.mice[mouse_id][cue_group].bins.bias_bins = bias_bins
                data_sets.mice[mouse_id][cue_group].bins.bias_incorrect_bins = bias_incorrect_bins
                data_sets.mice[mouse_id][cue_group].hit_rate = mouse_hit_rate
                data_sets.mice[mouse_id][cue_group].bias = mouse_bias
                data_sets.mice[mouse_id][cue_group].bias_incorrect = mouse_bias_incorrect
                data_sets.mice[mouse_id][cue_group].bias_corrected = mouse_bias_corrected.tolist()
                data_sets.mice[mouse_id][cue_group].dprime = mouse_dprime
                data_sets.mice[mouse_id][cue_group].bins.signal_detection_bins = signal_detection_bins

                # Store individual mouse data for plotting if requested
                if plot_individual_mice:
                    if mouse_id not in mouse_data_dict[cue_group]:
                        mouse_data_dict[cue_group][mouse_id] = []
                    
                    if plot_type == 'dprime':
                        mouse_data_dict[cue_group][mouse_id] = mouse_dprime
                    else:
                        mouse_data_dict[cue_group][mouse_id] = mouse_hit_rate

            # Compute across-mice statistics (normal flow)
            # Convert each measure to an array [mouse x angle_bin]
            hit_rate_array = np.array([
                data_sets.mice[m][cue_group].hit_rate for m in data_sets.mice
            ])
            bias_array = np.array([
                data_sets.mice[m][cue_group].bias for m in data_sets.mice
            ])
            bias_incorrect_array = np.array([
                data_sets.mice[m][cue_group].bias_incorrect for m in data_sets.mice
            ])
            bias_corrected_array = np.array([
                data_sets.mice[m][cue_group].bias_corrected for m in data_sets.mice
            ])
            dprime_array = np.array([
                data_sets.mice[m][cue_group].dprime for m in data_sets.mice
            ])

            n_mice = len(data_sets.mice)

            # Means + spreads
            mean_hit_rate = hit_rate_array.mean(axis=0)
            sem_hit_rate = hit_rate_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias = bias_array.mean(axis=0)
            sem_bias = bias_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias_incorrect = bias_incorrect_array.mean(axis=0)
            sem_bias_incorrect = bias_incorrect_array.std(axis=0) / np.sqrt(n_mice)

            mean_bias_corrected = bias_corrected_array.mean(axis=0)
            sem_bias_corrected = bias_corrected_array.std(axis=0) / np.sqrt(n_mice)
            
            mean_dprime = dprime_array.mean(axis=0)
            sem_dprime = dprime_array.std(axis=0) / np.sqrt(n_mice)

            # Fill in plotting_data with aggregated stats
            # Sorted by bin_start in ascending order
            bin_list = sorted(list(list(data_sets.mice.values())[0][cue_group].bins.hit_rate_bins.keys()))
            plotting_data.hit_rate = mean_hit_rate.tolist()
            plotting_data.hit_rate_sd = hit_rate_array.std(axis=0).tolist()
            plotting_data.hit_rate_sem = sem_hit_rate.tolist()

            plotting_data.bias = mean_bias.tolist()
            plotting_data.bias_sd = bias_array.std(axis=0).tolist()
            plotting_data.bias_sem = sem_bias.tolist()

            plotting_data.bias_incorrect = mean_bias_incorrect.tolist()
            plotting_data.bias_incorrect_sd = bias_incorrect_array.std(axis=0).tolist()
            plotting_data.bias_incorrect_sem = sem_bias_incorrect.tolist()

            plotting_data.bias_corrected = mean_bias_corrected.tolist()
            plotting_data.bias_corrected_sd = bias_corrected_array.std(axis=0).tolist()
            plotting_data.bias_corrected_sem = sem_bias_corrected.tolist()
            
            plotting_data.dprime = mean_dprime.tolist()
            plotting_data.dprime_sd = dprime_array.std(axis=0).tolist()
            plotting_data.dprime_sem = sem_dprime.tolist()

            plotting_data.n = n_mice
            plotting_data.bin_titles = [f"{b + (bin_size / 2):.2f}" for b in bin_list]

            # Plotting
            angles_deg = np.array(plotting_data.bin_titles, dtype=np.float64)

            if plot_mode == 'radial':
                # Convert angles to [0..360), then wrap
                adjusted_angles_deg = angles_deg % 360
                angles_rad = np.radians(adjusted_angles_deg)
                # Append first value to the end to close the loop visually
                angles_rad = np.append(angles_rad, angles_rad[0])

                if plot_type == 'hit_rate':
                    plot_data = np.append(plotting_data.hit_rate, plotting_data.hit_rate[0])
                    error_data = np.append(plotting_data.hit_rate_sem, plotting_data.hit_rate_sem[0])
                elif plot_type == 'bias_corrected':
                    plot_data = np.append(plotting_data.bias_corrected, plotting_data.bias_corrected[0])
                    error_data = np.append(plotting_data.bias_corrected_sem, plotting_data.bias_corrected_sem[0])
                elif plot_type == 'bias_incorrect':
                    plot_data = np.append(plotting_data.bias_incorrect, plotting_data.bias_incorrect[0])
                    error_data = np.append(plotting_data.bias_incorrect_sem, plotting_data.bias_incorrect_sem[0])
                elif plot_type == 'dprime':
                    plot_data = np.append(plotting_data.dprime, plotting_data.dprime[0])
                    error_data = np.append(plotting_data.dprime_sem, plotting_data.dprime_sem[0])
                elif plot_type == 'bias':
                    plot_data = np.append(plotting_data.bias, plotting_data.bias[0])
                    error_data = np.append(plotting_data.bias_sem, plotting_data.bias_sem[0])
                else:
                    raise ValueError(f"Unknown plot_type: {plot_type}")
                
                # Pick colour for the line
                if isinstance(sessions_input, dict) and len(sessions_dict) > 1:
                    # Multiple session groups: use smart colour assignment based on group names
                    dataset_name_lower = dataset_name.lower()
                    if 'ctrl' in dataset_name_lower or 'control' in dataset_name_lower:
                        colour = colors['all_trials']  # Blue
                    elif 'test' in dataset_name_lower:
                        colour = colors['visual_trials']  # Pink
                    else:
                        colour = colours_list[list(sessions_dict.keys()).index(dataset_name)]
                else:
                    # Original logic for single groups or multiple cue modes
                    colour = colors.get(cue_group, (0.5, 0.5, 0.5)) \
                        if len(cue_modes) > 1 else colours_list[list(sessions_dict.keys()).index(dataset_name)]

                # Decide label based on input type
                if isinstance(sessions_input, dict) and len(sessions_dict) > 1:
                    # Multiple session groups: use dataset name as label
                    label = dataset_name
                elif len(cue_modes) > 1:
                    # Multiple cue modes: use cue group as label
                    label = cue_group
                else:
                    # Single session group, single cue mode: use cue group as label
                    label = cue_group

                # Plot
                ax.plot(angles_rad, plot_data, marker='o', label=label, color=colour, linewidth=2)
                if error_bars == 'SEM':
                    ax.fill_between(angles_rad, plot_data - error_data, plot_data + error_data,
                                    alpha=0.3, color=lighten_colour(colour))

                # Plot individual mouse data if requested
                if plot_individual_mice:
                    for mouse, mouse_data in mouse_data_dict[cue_group].items():
                        # Choose the right data type based on plot_type
                        if plot_type == 'hit_rate':
                            individual_data = data_sets.mice[mouse][cue_group].hit_rate
                        elif plot_type == 'bias_corrected':
                            individual_data = data_sets.mice[mouse][cue_group].bias_corrected
                        elif plot_type == 'bias_incorrect':
                            individual_data = data_sets.mice[mouse][cue_group].bias_incorrect
                        elif plot_type == 'dprime':
                            individual_data = data_sets.mice[mouse][cue_group].dprime
                        elif plot_type == 'bias':
                            individual_data = data_sets.mice[mouse][cue_group].bias
                        else:
                            raise ValueError(f"Unknown plot_type: {plot_type}")
                        
                        # Close the circular plot for individual mouse data
                        individual_data_closed = np.append(individual_data, individual_data[0])
                        ax.plot(angles_rad, individual_data_closed, 
                               label=f"Mouse {mouse}", linestyle='--', marker='o', alpha=0.7)

                # Plot circular mean vectors if requested (not for d-prime)
                if plot_circular_means and show_circular_stats and plot_type != 'dprime' and dataset_name in circular_stats_by_group:
                    group_stats = circular_stats_by_group[dataset_name][cue_group]
                    
                    # Calculate overall group circular mean from individual mouse means
                    all_valid_means = [m for m in group_stats.mouse_means if not np.isnan(m)]
                    if all_valid_means:
                        group_circ_stats = calculate_circular_statistics(
                            all_valid_means, 
                            np.ones(len(all_valid_means))
                        )
                        
                        # Plot group mean arrow using custom function
                        mean_rad = np.radians(group_circ_stats['circular_mean'])
                        y_max = ax.get_ylim()[1]
                        arrow_length = group_circ_stats['resultant_length'] * y_max * 0.7
                        
                        # Draw main arrow
                        draw_polar_arrow(ax, mean_rad, arrow_length, colour, 
                                       linewidth=3, alpha=0.8)
                        
                        # Add to legend
                        ax.plot([], [], color=colour, linewidth=3, 
                               label=f"{dataset_name} mean")
                        
                        # Optionally plot individual mouse means as smaller arrows
                        if plot_individual_mice:
                            for i, (mouse_id, mean_angle, resultant) in enumerate(
                                    zip(group_stats.mouse_ids, 
                                        group_stats.mouse_means, 
                                        group_stats.mouse_resultants)):
                                
                                if not np.isnan(mean_angle):
                                    mean_rad = np.radians(mean_angle)
                                    arrow_length = resultant * y_max * 0.5
                                    
                                    # Draw smaller arrow
                                    draw_polar_arrow(ax, mean_rad, arrow_length,
                                                   lighten_colour(colour, 0.5),
                                                   linewidth=1.5, alpha=0.5,
                                                   head_width=0.08, head_length=0.06)

            elif plot_mode == 'linear_comparison':
                # Decide which metric to plot
                if plot_type == 'hit_rate':
                    plot_data = plotting_data.hit_rate
                    error_data = plotting_data.hit_rate_sem
                elif plot_type == 'bias_corrected':
                    plot_data = plotting_data.bias_corrected
                    error_data = plotting_data.bias_corrected_sem
                elif plot_type == 'bias_incorrect':
                    plot_data = plotting_data.bias_incorrect
                    error_data = plotting_data.bias_incorrect_sem
                elif plot_type == 'dprime':
                    plot_data = plotting_data.dprime
                    error_data = plotting_data.dprime_sem
                else:  # 'bias'
                    plot_data = plotting_data.bias
                    error_data = plotting_data.bias_sem

                # Pick colour for the line
                if isinstance(sessions_input, dict) and len(sessions_dict) > 1:
                    # Multiple session groups: use smart colour assignment based on group names
                    dataset_name_lower = dataset_name.lower()
                    if 'ctrl' in dataset_name_lower or 'control' in dataset_name_lower:
                        colour = colors['all_trials']  # Blue
                    elif 'test' in dataset_name_lower:
                        colour = colors['visual_trials']  # Pink
                    else:
                        colour = colours_list[list(sessions_dict.keys()).index(dataset_name)]
                else:
                    # Original logic for single groups or multiple cue modes
                    colour = colors.get(cue_group, (0.5, 0.5, 0.5)) \
                        if len(cue_modes) > 1 else colours_list[list(sessions_dict.keys()).index(dataset_name)]

                # Decide label based on input type
                if isinstance(sessions_input, dict) and len(sessions_dict) > 1:
                    # Multiple session groups: use dataset name as label
                    label = dataset_name
                elif len(cue_modes) > 1:
                    # Multiple cue modes: use cue group as label
                    label = cue_group
                else:
                    # Single session group, single cue mode: use cue group as label
                    label = cue_group

                # Plot
                ax.plot(angles_deg, plot_data, marker='o', label=label, color=colour, linewidth=2)
                if error_bars == 'SEM':
                    ax.fill_between(angles_deg, 
                                    np.array(plot_data) - np.array(error_data), 
                                    np.array(plot_data) + np.array(error_data),
                                    alpha=0.3, color=lighten_colour(colour))

                # Plot individual mouse data if requested
                if plot_individual_mice:
                    for mouse, mouse_data in mouse_data_dict[cue_group].items():
                        # Choose the right data type based on plot_type
                        if plot_type == 'hit_rate':
                            individual_data = mouse_data
                        elif plot_type == 'bias_corrected':
                            individual_data = data_sets.mice[mouse][cue_group].bias_corrected
                        elif plot_type == 'bias_incorrect':
                            individual_data = data_sets.mice[mouse][cue_group].bias_incorrect
                        elif plot_type == 'dprime':
                            individual_data = data_sets.mice[mouse][cue_group].dprime
                        else:  # 'bias'
                            individual_data = data_sets.mice[mouse][cue_group].bias
                        
                        ax.plot(angles_deg, individual_data, 
                               label=f"Mouse {mouse}", linestyle='--', marker='o', alpha=0.7)

    # Final plot cosmetics
    # Position legend outside the plot area to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=True, 
              fancybox=True, shadow=True)
    ax.set_title(plot_title)
    
    if plot_mode == 'radial':
        # Set y-limits based on plot type
        if plot_type == 'dprime':
            # D-prime has different scale
            ax.set_ylim(-2, 5)  # Typical d-prime range
        elif plot_type == 'bias' or plot_type == 'bias_incorrect' or plot_type == 'bias_corrected':
            ax.set_ylim(0, max(plot_data) * 1.1 if len(plot_data) else 1)
        else:
            ax.set_ylim(0, 1)  # For standard hit_rate

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_xlim(np.radians(limits[0]), np.radians(limits[1]))

        # Create angle labels
        angles_label = np.arange(-180, 181, 30)
        # Avoid overlapping +180 / -180
        if len(angles_label) > 1 and angles_label[-1] == 180:
            angles_label = angles_label[:-1]

        ax.set_xticks(np.radians(angles_label))
        ax.set_xticklabels([f'{int(a)}°' for a in angles_label])
        ax.grid(True)

    elif plot_mode == 'linear_comparison':
        ax.set_xlabel(x_title or 'Turn Angle (degrees)')
        
        # Set y-axis label based on plot type
        if plot_type == 'dprime':
            ax.set_ylabel(y_title or "d' (Sensitivity)")
        elif plot_type in ('bias', 'bias_incorrect'):
            ax.set_ylabel(y_title or 'Bias')
        elif plot_type == 'bias_corrected':
            ax.set_ylabel(y_title or 'Bias-Corrected Hit Rate')
        else:
            ax.set_ylabel(y_title or 'Hit Rate')
            
        ax.set_xlim(limits[0], limits[1])

        # Rescale Y axis if needed
        if plot_type == 'dprime':
            ax.set_ylim(-2, 5)  # Typical d-prime range
        elif plot_type == 'bias_corrected':
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

    # Print circular statistics if calculated
    # Print circular statistics if calculated (not for d-prime)
    if show_circular_stats and circular_stats_by_group and plot_type != 'dprime':
        # Call the utility function
        stats_summary = print_circular_statistics_summary(
            circular_stats_by_group=circular_stats_by_group,
            cue_modes=cue_modes
        )

    plt.show()