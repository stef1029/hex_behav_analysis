from hex_behav_analysis.debug.zz_scales_alignment_core import ScalesAlignmentCore

"""
Script to align scales activation times with PC log trial times.
This module provides visualization and analysis functionality.
Modified to use linear transformations at all stages.
"""

import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import socket


class ScalesTrialAligner(ScalesAlignmentCore):
    """
    Align scales platform events with PC log trials with advanced visualization.
    Inherits core functionality from ScalesAlignmentCore.
    Modified to use linear transformations throughout all alignment stages.
    """
    
    def __init__(self, session_dict):
        """
        Initialise the aligner with session information.
        
        Args:
            session_dict (dict): Session dictionary from Cohort_folder
        """
        super().__init__(session_dict)
    
    def plot_alignment(self, results, output_path=None, web_display=False, initial_matches=None):
        """
        Plot the alignment results with timing difference histogram.
        Shows before/after alignment and distribution of timing errors.
        
        Args:
            results (dict): Results from align_and_match
            output_path (str): Path to save plot (optional)
            web_display (bool): Whether to display in web browser
            initial_matches (list): Initial matched trials from Stage 1 to show with final transformation
        """
        # Configure matplotlib backend for web display if requested
        if web_display:
            matplotlib.use('webagg')
            matplotlib.pyplot.rcParams['webagg.open_in_browser'] = False
            
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[1, 1, 1, 1])
        
        # Handle both simple offset and linear transformation
        if 'scale' in results:
            offset_text = f"scale: {results['scale']:.6f}, offset: {results['offset']:.3f}s"
            transform_func = lambda t: t * results['scale'] + results['offset']
        else:
            offset_text = f"offset: {results['offset']:.3f}s"
            transform_func = lambda t: t + results['offset']
        
        # Top plot: Scales trace (full session)
        weights = self.scales_data['data']
        timestamps = self.scales_data['timestamps']
        
        # Use full data instead of time window
        plot_weights = weights
        plot_times = timestamps
        
        ax1.plot(plot_times, plot_weights, 'b-', linewidth=0.5, alpha=0.8, label='Weight')
        ax1.axhline(y=self.mouse_weight_threshold, color='r', linestyle='--', 
                    alpha=0.7, linewidth=2, label=f'Threshold ({self.mouse_weight_threshold}g)')
        
        # Shade platform events
        above_threshold = np.array(plot_weights) >= self.mouse_weight_threshold
        if np.any(above_threshold):
            diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                if start < len(plot_times) and end <= len(plot_times):
                    duration = plot_times[end-1] - plot_times[start]
                    if duration >= 1.0:
                        ax1.axvspan(plot_times[start], plot_times[end-1], alpha=0.3, color='green')
        
        ax1.set_ylabel('Weight (g)')
        ax1.set_title('Scales Trace (Full Session)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        # No xlim set - shows full range
        
        # Second plot: Original times before alignment
        ax2.scatter(self.platform_events, np.ones(len(self.platform_events)), 
                   c='blue', s=100, label='Platform events', marker='o')
        ax2.scatter(self.pc_trials, np.zeros(len(self.pc_trials)), 
                   c='red', s=100, label='PC trials', marker='s')
        
        # Add refinement info to title if available
        if 'refinement_info' in results:
            title = f'Original Times (Refinement used {results["refinement_info"]["n_pairs_used_for_fit"]} matched pairs)'
        else:
            title = 'Original Times (Before Alignment)'
        ax2.set_title(title)
        
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_ylabel('Event Type')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['PC Trials', 'Platform Events'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Third plot: Aligned times showing all data after transformation
        # Get adjusted PC times from results
        adjusted_pc_times = []
        for trial in results['matched_pc_trials'] + results['unmatched_pc_trials']:
            adjusted_pc_times.append(trial['adjusted_time'])
        
        # Plot matched connections first (so they appear behind scatter points)
        for match in results['matched_pc_trials']:
            ax3.plot([match['adjusted_time'], match['platform_time']], [0, 1], 
                    'g-', alpha=0.5, linewidth=1)
        
        # Plot all events
        ax3.scatter(self.platform_events, np.ones(len(self.platform_events)), 
                   c='blue', s=100, label='Platform events', marker='o', zorder=3)
        ax3.scatter(adjusted_pc_times, np.zeros(len(adjusted_pc_times)), 
                   c='red', s=100, label=f'PC trials ({offset_text})', marker='s', zorder=3)
        
        # Highlight unmatched events
        unmatched_pc_times = [e['adjusted_time'] for e in results['unmatched_pc_trials']]
        if unmatched_pc_times:
            ax3.scatter(unmatched_pc_times, np.zeros(len(unmatched_pc_times)), 
                       c='orange', s=150, label='Unmatched PC trials', marker='s', 
                       edgecolors='black', linewidth=2, zorder=4)
        
        unmatched_platform_times = [e['platform_time'] for e in results['unmatched_platform_events']]
        if unmatched_platform_times:
            ax3.scatter(unmatched_platform_times, np.ones(len(unmatched_platform_times)), 
                       c='orange', s=150, label='Unmatched platform', marker='o', 
                       edgecolors='black', linewidth=2, zorder=4)
        
        ax3.set_title(f'After Alignment - PC Match Rate: {results["pc_match_rate"]*100:.1f}% '
                     f'(threshold: {results["max_match_distance"]}s)')
        
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_ylabel('Event Type')
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['PC Trials', 'Platform Events'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Bottom plot: Histogram of timing differences
        # If initial_matches provided, use those with current transformation
        if initial_matches is not None:
            # Apply current transformation to initial matches
            differences = []
            for match in initial_matches:
                pc_time = match['pc_trial_time']
                platform_time = match['platform_time']
                adjusted_time = transform_func(pc_time)
                distance = abs(adjusted_time - platform_time)
                differences.append(distance)
            
            ax4.hist(differences, bins=30, alpha=0.7, edgecolor='black', 
                    label=f'Stage 1 matches ({len(initial_matches)} trials)')
            ax4.set_title('Distribution of Timing Differences (Stage 1 Matches with Final Transformation)')
        else:
            # Use current matches
            differences = [match['distance'] for match in results['matched_pc_trials']]
            ax4.hist(differences, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_title('Distribution of Timing Differences After Alignment')
        
        ax4.set_xlabel('Timing Difference (seconds)')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # Add vertical lines for statistics
        if differences:
            mean_diff = np.mean(differences)
            median_diff = np.median(differences)
            ax4.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.3f}s')
            ax4.axvline(median_diff, color='green', linestyle='--', label=f'Median: {median_diff:.3f}s')
            ax4.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        if web_display:
            self._show_web_display()
            
        plt.show()
        
        if not web_display:
            plt.close()
    
    def plot_signed_differences(self, results, output_path=None, web_display=False):
        """
        Plot histogram of signed timing differences to show if PC trials are early or late.
        Provides insight into systematic timing bias.
        
        Args:
            results (dict): Results from align_and_match
            output_path (str): Path to save plot
            web_display (bool): Whether to display in web browser
        """
        if not results['matched_pc_trials']:
            print("No matched trials to plot")
            return
        
        # Configure matplotlib backend for web display if requested
        if web_display:
            matplotlib.use('webagg')
            matplotlib.pyplot.rcParams['webagg.open_in_browser'] = False
            
        # Extract signed differences
        signed_differences = []
        for match in results['matched_pc_trials']:
            signed_diff = match['adjusted_time'] - match['platform_time']
            signed_differences.append(signed_diff)
        
        signed_differences = np.array(signed_differences)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(signed_differences, bins=50, alpha=0.7, edgecolor='black')
        
        # Colour bars based on sign
        for i, patch in enumerate(patches):
            if bins[i] + (bins[i+1] - bins[i])/2 < 0:
                patch.set_facecolor('blue')  # PC early
            else:
                patch.set_facecolor('red')   # PC late
        
        # Add vertical lines for statistics
        mean_diff = np.mean(signed_differences)
        median_diff = np.median(signed_differences)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Perfect alignment')
        ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_diff:.3f}s')
        ax.axvline(median_diff, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_diff:.3f}s')
        
        # Labels and title
        ax.set_xlabel('Timing Difference (PC time - Platform time) [seconds]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Signed Timing Differences After Alignment\n'
                    '(Negative = PC early, Positive = PC late)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with statistics
        stats_text = (f'PC Early: {np.sum(signed_differences < 0)} trials\n'
                     f'PC Late: {np.sum(signed_differences > 0)} trials\n'
                     f'Mean offset: {mean_diff:.3f}s\n'
                     f'Std dev: {np.std(signed_differences):.3f}s')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Signed differences plot saved to: {output_path}")
        
        if web_display:
            self._show_web_display()
            
        plt.show()
        
        if not web_display:
            plt.close()
    
    def plot_scales_trace_with_trials(self, results, time_window=None, output_path=None, web_display=False):
        """
        Plot the full scales trace showing threshold crossings and aligned PC trial times.
        
        Args:
            results (dict): Results from align_and_match containing the offset
            time_window (tuple): Optional (start, end) times to zoom in on
            output_path (str): Path to save plot (optional)
            web_display (bool): Whether to display in web browser
        """
        # Configure matplotlib backend for web display if requested
        if web_display:
            matplotlib.use('webagg')
            matplotlib.pyplot.rcParams['webagg.open_in_browser'] = False
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Get scales data
        weights = self.scales_data['data']
        timestamps = self.scales_data['timestamps']
        
        # Apply time window if specified
        if time_window:
            start_time, end_time = time_window
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            plot_weights = weights[mask]
            plot_times = timestamps[mask]
        else:
            plot_weights = weights
            plot_times = timestamps
            
        # Top subplot: Raw scales trace
        ax1.plot(plot_times, plot_weights, 'b-', linewidth=0.5, alpha=0.8, label='Weight')
        ax1.axhline(y=self.mouse_weight_threshold, color='r', linestyle='--', 
                    alpha=0.7, linewidth=2, label=f'Threshold ({self.mouse_weight_threshold}g)')
        
        # Shade regions where weight is above threshold
        above_threshold = np.array(plot_weights) >= self.mouse_weight_threshold
        
        # Find continuous regions above threshold
        if np.any(above_threshold):
            # Get start and end indices of continuous regions
            diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Shade regions
            platform_label_added = False
            short_label_added = False
            for start, end in zip(starts, ends):
                if start < len(plot_times) and end <= len(plot_times):
                    duration = plot_times[end-1] - plot_times[start]
                    if duration >= 1.0:
                        # Platform event (≥1s)
                        label = 'Platform event' if not platform_label_added else ''
                        ax1.axvspan(plot_times[start], plot_times[end-1], alpha=0.3, color='green', label=label)
                        platform_label_added = True
                        # Mark trial start point (1s after platform entry)
                        event_time = plot_times[start] + 1.0
                        ax1.axvline(x=event_time, color='darkgreen', linestyle=':', 
                                   alpha=0.7, linewidth=1)
                    else:
                        # Short activation
                        label = 'Short activation' if not short_label_added else ''
                        ax1.axvspan(plot_times[start], plot_times[end-1], alpha=0.2, color='yellow', label=label)
                        short_label_added = True
        
        ax1.set_ylabel('Weight (g)', fontsize=12)
        ax1.set_title('Scales Trace with Platform Events', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Event alignment
        # Plot platform events
        platform_in_window = [t for t in self.platform_events 
                            if (not time_window or (time_window[0] <= t <= time_window[1]))]
        ax2.scatter(platform_in_window, np.ones(len(platform_in_window)) * 2, 
                c='green', s=100, marker='o', label='Platform events', zorder=3)
        
        # Plot aligned PC trial times
        matched_label_added = False
        unmatched_label_added = False
        
        for match in results['matched_pc_trials']:
            if not time_window or (time_window[0] <= match['platform_time'] <= time_window[1]):
                label = 'Matched PC trials' if not matched_label_added else ''
                ax2.scatter([match['adjusted_time']], [1], c='red', s=100, marker='s', 
                        label=label, zorder=3)
                matched_label_added = True
                # Draw connection line
                ax2.plot([match['adjusted_time'], match['platform_time']], [1, 2], 
                        'g-', alpha=0.5, linewidth=1, zorder=1)
        
        # Plot unmatched events
        for event in results['unmatched_platform_events']:
            platform_time = event['platform_time']
            if not time_window or (time_window[0] <= platform_time <= time_window[1]):
                ax2.scatter([platform_time], [2], c='orange', s=150, marker='o', 
                        edgecolors='black', linewidth=2, zorder=4)
        
        for trial in results['unmatched_pc_trials']:
            adjusted_time = trial['adjusted_time']
            if not time_window or (time_window[0] <= adjusted_time <= time_window[1]):
                label = 'Unmatched' if not unmatched_label_added else ''
                ax2.scatter([adjusted_time], [1], c='orange', s=150, marker='s', 
                        edgecolors='black', linewidth=2, label=label, zorder=4)
                unmatched_label_added = True
        
        ax2.set_ylim(0.5, 2.5)
        ax2.set_yticks([1, 2])
        ax2.set_yticklabels(['PC trials (aligned)', 'Platform events'])
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_title(f'Event Alignment Visualization (PC match rate: {results["pc_match_rate"]*100:.1f}%)', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis limits
        if time_window:
            plt.xlim(time_window)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        if web_display:
            self._show_web_display()
            
        plt.show()
        
        if not web_display:
            plt.close()
    
    def _show_web_display(self):
        """Helper method to display web server information."""
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        port = 8988  # Default WebAgg port
        
        print("\n" + "="*50)
        print("Web server started!")
        print(f"Access the plot at: http://{local_ip}:{port}")
        print(f"Or locally at: http://localhost:{port}")
        print("Press Ctrl+C in the terminal to stop the server")
        print("="*50 + "\n")
    
    def run_three_stage_alignment(self, plot=True, output_dir=None, web_display=False):
        """
        Run a three-stage alignment process with comprehensive visualization:
        1. Initial alignment with linear transformation
        2. Refined alignment using matched pairs with linear transformation
        3. Final refinement using only best matches
        
        All stages now use linear transformations instead of simple offsets.
        
        Args:
            plot (bool): Whether to plot results
            output_dir (str): Directory for output plots
            web_display (bool): Whether to display plots in web browser
            
        Returns:
            tuple: (initial_results, refined_results, final_results, aligned_trials)
        """
        print("\n" + "="*60)
        print("THREE-STAGE LINEAR ALIGNMENT PROCESS")
        print("="*60)
        
        # Load data
        print("\nLoading scales data...")
        self.load_scales_data()
        
        # Get platform events
        platform_times = self.get_platform_events(min_duration=1.0)
        print(f"Found {len(platform_times)} platform events (≥1s duration)")
        
        # Load PC trials
        print("\nLoading PC trials...")
        trials = self.load_pc_trials()
        print(f"Found {len(trials)} PC trials")
        
        # Show breakdown by port
        port_counts = {}
        for trial in trials:
            port = trial.get('port', 'unknown')
            port_counts[port] = port_counts.get(port, 0) + 1
        
        print("\nTrials by port:")
        for port in sorted(port_counts.keys()):
            print(f"  Port {port}: {port_counts[port]}")
        
        # Stage 1: Initial linear alignment
        print("\n" + "-"*60)
        print("STAGE 1: Initial linear alignment")
        print("-"*60)
        
        # Find initial linear transformation
        print("Finding initial linear transformation...")
        best_scale, best_offset = self._find_initial_linear_transform(max_match_distance=2.0)
        
        print(f"Initial linear transformation:")
        print(f"  Scale: {best_scale:.8f}")
        print(f"  Offset: {best_offset:.3f}s")
        print(f"  Clock drift: {(best_scale-1)*3600*1000:.3f} ms/hour")
        
        # Apply initial linear transformation
        initial_results = self.align_and_match_linear(best_scale, best_offset, max_match_distance=2.0)
        initial_stats = self.analyze_timing_differences(initial_results)
        initial_results['timing_stats'] = initial_stats
        
        print(f"\nInitial Linear Alignment Results:")
        print(f"  Matched PC trials: {len(initial_results['matched_pc_trials'])}/{len(self.pc_trials)} ({initial_results['pc_match_rate']*100:.1f}%)")
        print(f"  Using threshold: {initial_results['max_match_distance']}s (loose for learning)")
        if initial_stats['mean'] is not None:
            print(f"  Mean timing difference: {initial_stats['mean']*1000:.1f}ms")
            print(f"  RMS error: {initial_stats['rms']*1000:.1f}ms")
        
        # Stage 2: Refined alignment
        print("\n" + "-"*60)
        print("STAGE 2: Refined alignment with matched pairs")
        print("-"*60)
        
        refined_results = self.refine_alignment_with_matches(initial_results)
        refined_stats = self.analyze_timing_differences(refined_results)
        refined_results['timing_stats'] = refined_stats
        
        print(f"\nRefined Alignment Results:")
        print(f"  Matched PC trials: {len(refined_results['matched_pc_trials'])}/{len(self.pc_trials)} ({refined_results['pc_match_rate']*100:.1f}%)")
        if refined_stats['mean'] is not None:
            print(f"  Mean timing difference: {refined_stats['mean']*1000:.1f}ms")
            print(f"  RMS error: {refined_stats['rms']*1000:.1f}ms")
        
        # Stage 3: Final refinement with best matches
        print("\n" + "-"*60)
        print("STAGE 3: Final refinement with best matches")
        print("-"*60)
        
        final_results = self.refine_with_best_matches(refined_results, threshold_ms=80)
        final_stats = self.analyze_timing_differences(final_results)
        final_results['timing_stats'] = final_stats
        
        print(f"\nFinal Alignment Results (Stage 3 refinement):")
        print(f"  Matched PC trials: {len(final_results['matched_pc_trials'])}/{len(self.pc_trials)} ({final_results['pc_match_rate']*100:.1f}%)")
        print(f"  Using threshold: {final_results['max_match_distance']}s (tight for final refinement)")
        
        if final_stats['mean'] is not None:
            print(f"\nFinal Timing Statistics (Stage 3 matches only):")
            print(f"  Mean difference: {final_stats['mean']*1000:.1f}ms")
            print(f"  Std deviation: {final_stats['std']*1000:.1f}ms")
            print(f"  Median difference: {final_stats['median']*1000:.1f}ms")
            print(f"  RMS error: {final_stats['rms']*1000:.1f}ms")
            print(f"  Range: [{final_stats['min']*1000:.1f}, {final_stats['max']*1000:.1f}]ms")
            
            print(f"\nMatches within thresholds (Stage 3 only):")
            for threshold, count in final_stats['within_thresholds'].items():
                percentage = (count / final_stats['count']) * 100 if final_stats['count'] > 0 else 0
                print(f"  Within {threshold}: {count}/{final_stats['count']} ({percentage:.1f}%)")
        
        # Get aligned trial times with final 2ms correction
        aligned_trials = self.get_aligned_trial_times(final_results, final_correction=0.002)
        
        # Apply final transformation to Stage 1 matches for reporting
        print("\n" + "="*60)
        print("STAGE 1 MATCHES WITH FINAL TRANSFORMATION")
        print("="*60)
        
        # Get transformation parameters from final results
        if 'scale' in final_results:
            final_scale = final_results['scale']
            final_offset = final_results['offset']
        else:
            final_scale = 1.0
            final_offset = final_results['offset']
        
        # Re-evaluate Stage 1 matches with final transformation
        stage1_with_final = []
        for match in initial_results['matched_pc_trials']:
            pc_time = match['pc_trial_time']
            platform_time = match['platform_time']
            # Apply final transformation plus 2ms correction
            final_adjusted_time = pc_time * final_scale + final_offset + 0.002
            final_distance = abs(final_adjusted_time - platform_time)
            stage1_with_final.append({
                'pc_trial_time': pc_time,
                'platform_time': platform_time,
                'adjusted_time': final_adjusted_time,
                'distance': final_distance,
                'trial_idx': match['trial_idx']
            })
        
        # Calculate statistics for Stage 1 matches with final transformation
        stage1_final_distances = [m['distance'] for m in stage1_with_final]
        stage1_final_stats = {
            'mean': np.mean(stage1_final_distances),
            'std': np.std(stage1_final_distances),
            'median': np.median(stage1_final_distances),
            'rms': np.sqrt(np.mean(np.array(stage1_final_distances)**2)),
            'within_250ms': np.sum(np.array(stage1_final_distances) <= 0.25),
            'within_500ms': np.sum(np.array(stage1_final_distances) <= 0.5),
        }
        
        print(f"Original Stage 1 matches: {len(initial_results['matched_pc_trials'])} (with 2s threshold)")
        print(f"Performance with final transformation:")
        print(f"  Mean error: {stage1_final_stats['mean']*1000:.1f}ms")
        print(f"  Std deviation: {stage1_final_stats['std']*1000:.1f}ms")
        print(f"  Median error: {stage1_final_stats['median']*1000:.1f}ms")
        print(f"  RMS error: {stage1_final_stats['rms']*1000:.1f}ms")
        print(f"  Within 250ms: {stage1_final_stats['within_250ms']}/{len(stage1_with_final)} ({stage1_final_stats['within_250ms']/len(stage1_with_final)*100:.1f}%)")
        print(f"  Within 500ms: {stage1_final_stats['within_500ms']}/{len(stage1_with_final)} ({stage1_final_stats['within_500ms']/len(stage1_with_final)*100:.1f}%)")
        
        # Store Stage 1 matches with final transformation in results
        final_results['stage1_matches_final_transform'] = stage1_with_final
        final_results['stage1_final_stats'] = stage1_final_stats
        
        # Print summary of transformations
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY")
        print("="*60)
        print(f"Stage 1 - Initial linear transformation:")
        print(f"  Scale: {initial_results['scale']:.8f}")
        print(f"  Offset: {initial_results['offset']:.3f}s")
        print(f"  Clock drift: {(initial_results['scale']-1)*3600*1000:.3f} ms/hour")
        if 'refinement_info' in final_results:
            print(f"Stage 2 - Refined linear transformation:")
            print(f"  Scale: {final_results['refinement_info']['refined_scale']:.8f}")
            print(f"  Offset: {final_results['refinement_info']['refined_offset']:.3f}s")
            print(f"  Clock drift: {(final_results['refinement_info']['refined_scale']-1)*3600*1000:.3f} ms/hour")
        if 'final_refinement_info' in final_results:
            print(f"Stage 3 - Final transformation:")
            print(f"  Scale: {final_results['final_refinement_info']['final_scale']:.8f}")
            print(f"  Offset: {final_results['final_refinement_info']['final_offset']:.3f}s")
            print(f"  Clock drift: {(final_results['final_refinement_info']['final_scale']-1)*3600*1000:.3f} ms/hour")
            print(f"  Final correction: +2ms (empirical adjustment)")
        
        # Print sample of aligned trial data
        print("\n" + "="*60)
        print("ALIGNED TRIAL DATA (first 10 trials)")
        print("="*60)
        print(f"{'Idx':>4} {'PC Time':>10} {'Aligned':>10} {'Port':>5} {'Outcome':>10} {'Matched':>8} {'Distance':>10}")
        print("-" * 60)
        
        for trial in aligned_trials[:10]:
            matched_str = "Yes" if trial['matched'] else "No"
            distance_str = f"{trial['match_distance']*1000:.1f}ms" if trial['match_distance'] is not None else "N/A"
            port_str = str(trial['port']) if trial['port'] is not None else "None"
            outcome_str = trial['outcome'] if trial['outcome'] is not None else "None"
            print(f"{trial['trial_idx']:>4} {trial['pc_time']:>10.3f} {trial['aligned_time']:>10.3f} "
                  f"{port_str:>5} {outcome_str:>10} {matched_str:>8} {distance_str:>10}")
        
        if len(aligned_trials) > 10:
            print(f"... ({len(aligned_trials) - 10} more trials)")
        
        # Plot if requested
        if plot:
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Plot initial alignment
                print("\nGenerating alignment visualizations...")
                initial_path = output_dir / f"alignment_initial_{self.session_id}.png"
                self.plot_alignment(initial_results, initial_path, web_display=False)
                
                # Plot refined alignment
                refined_path = output_dir / f"alignment_refined_{self.session_id}.png"
                self.plot_alignment(refined_results, refined_path, web_display=False)
                
                # Plot final alignment with Stage 1 matches
                final_path = output_dir / f"alignment_final_{self.session_id}.png"
                self.plot_alignment(final_results, final_path, web_display=web_display, 
                                  initial_matches=initial_results['matched_pc_trials'])
                
                # Plot signed differences histogram
                signed_path = output_dir / f"signed_differences_{self.session_id}.png"
                self.plot_signed_differences(final_results, signed_path, web_display=False)
                
                # Plot scales trace with final alignment
                trace_path = output_dir / f"scales_trace_{self.session_id}.png"
                self.plot_scales_trace_with_trials(
                    final_results,
                    time_window=(0, 600),  # First 10 minutes
                    output_path=trace_path,
                    web_display=False
                )
                
                print(f"All plots saved to: {output_dir}")
            else:
                # Display plots without saving
                self.plot_alignment(final_results, None, web_display=web_display,
                                  initial_matches=initial_results['matched_pc_trials'])
                self.plot_signed_differences(final_results, None, web_display=False)
                self.plot_scales_trace_with_trials(
                    final_results,
                    time_window=(0, 600),
                    output_path=None,
                    web_display=False
                )
        
        return initial_results, refined_results, final_results, aligned_trials
    
    def _find_initial_linear_transform(self, max_match_distance=2.0):
        """
        Find initial linear transformation using grid search optimization.
        
        Args:
            max_match_distance (float): Maximum distance for considering a match
            
        Returns:
            tuple: (best_scale, best_offset)
        """
        from scipy.optimize import differential_evolution
        
        platform_array = np.array(self.platform_events)
        pc_array = np.array(self.pc_trials)
        
        def objective(params):
            """Objective function: minimize sum of minimum distances."""
            scale, offset = params
            total_cost = 0
            
            # For each PC trial, find distance to nearest platform event
            for pc_time in pc_array:
                adjusted_time = pc_time * scale + offset
                distances = np.abs(platform_array - adjusted_time)
                min_distance = np.min(distances)
                # Use squared distance with threshold
                if min_distance <= max_match_distance:
                    total_cost += min_distance ** 2
                else:
                    total_cost += max_match_distance ** 2
            
            return total_cost
        
        # Set bounds for optimization
        # Scale: allow up to 0.5% clock drift (5000 ppm)
        scale_bounds = (0.995, 1.005)
        # Offset: search within ±100 seconds
        offset_bounds = (-100, 100)
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds=[scale_bounds, offset_bounds],
            seed=42,
            maxiter=100,
            popsize=15,
            atol=1e-10,
            tol=1e-10
        )
        
        return result.x[0], result.x[1]


# Example usage
if __name__ == "__main__":
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    
    print("Loading cohort info...")
    cohort = Cohort_folder('/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment', use_existing_cohort_info=True)
    
    # Get a specific session
    session_dict = cohort.get_session("250611_112601_mtao106-3a")
    
    # Create aligner
    aligner = ScalesTrialAligner(session_dict)
    
    # Run three-stage alignment with all visualizations
    # Now uses linear transformations at all stages
    initial_results, refined_results, final_results, aligned_trials = aligner.run_three_stage_alignment(
        plot=True,
        output_dir="/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/debug/alignment_plots",
        web_display=True  # Set to True to view plots in web browser
    )
    
    # The aligned_trials list now contains all PC trials with their aligned DAQ times
    print(f"\n\nTotal aligned trials available: {len(aligned_trials)}")