# Import the alignment class from your alignment script
# Adjust this import path as needed for your setup
from hex_behav_analysis.debug.trial_finder_scales_times_alignment import ScalesTrialAligner

"""
Script to compare Port 4 trial times from:
1. PC logs aligned using scales/platform events
2. Direct DAQ LED_4 trace detection

This serves as validation for the scales-based alignment method when LED traces are corrupted.
"""

import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy import stats as scipy_stats
from scipy.stats import linregress


class Port4TrialComparison:
    """
    Compare Port 4 trial times from aligned PC logs vs DAQ LED traces.
    """
    
    def __init__(self, session_dict):
        """
        Initialise the comparison with session information.
        
        Args:
            session_dict (dict): Session dictionary from Cohort_folder
        """
        self.session_dict = session_dict
        self.session_id = session_dict.get('session_id')
        
        # Load NWB file path
        if session_dict.get('portable'):
            self.nwb_file_path = Path(session_dict.get('NWB_file'))
        else:
            self.nwb_file_path = Path(session_dict.get('processed_data', {}).get('NWB_file'))
        
        # Data containers
        self.led4_data = None
        self.aligned_pc_trials = None
        self.daq_trials = []
        
    def load_led4_data(self):
        """
        Load LED_4 data from NWB file for DAQ-based detection.
        """
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            nwbfile = io.read()
            
            # Load LED_4 data
            if "LED_4" in nwbfile.stimulus:
                ts = nwbfile.stimulus["LED_4"]
                self.led4_data = {
                    'data': ts.data[:],
                    'timestamps': ts.timestamps[:]
                }
                print(f"Loaded LED_4 data: {len(self.led4_data['data'])} samples")
            else:
                raise ValueError("LED_4 channel not found in NWB file")
    
    def detect_daq_trials(self):
        """
        Detect Port 4 trials from LED_4 on/off events in DAQ trace.
        """
        data = self.led4_data['data']
        timestamps = self.led4_data['timestamps']
        
        # Check if LED is stuck high
        unique_values = np.unique(data)
        if len(unique_values) == 1 and unique_values[0] == 1:
            print("WARNING: LED_4 appears to be stuck high")
            return []
        
        # Find transitions (LED turning on)
        transitions = np.diff(np.concatenate(([0], data)))  # Prepend 0 to catch first high
        on_indices = np.where(transitions > 0)[0]
        
        # Extract trial start times (when LED turns on)
        trials = []
        for i, on_idx in enumerate(on_indices):
            # Find the corresponding off transition
            remaining_data = data[on_idx:]
            off_relative_idx = np.where(remaining_data == 0)[0]
            
            if len(off_relative_idx) > 0:
                off_idx = on_idx + off_relative_idx[0]
            else:
                # LED stays on until end
                off_idx = len(timestamps) - 1
            
            trial = {
                'trial_no': i,
                'led_on_time': timestamps[on_idx],
                'led_off_time': timestamps[off_idx],
                'duration': timestamps[off_idx] - timestamps[on_idx]
            }
            trials.append(trial)
        
        self.daq_trials = trials
        print(f"Detected {len(trials)} LED_4 trials from DAQ trace")
        return trials
    
    def get_aligned_pc_trials(self):
        """
        Get aligned trial times from PC logs using the ScalesTrialAligner.
        """
        print("\nRunning scales-based alignment for PC trials...")
        
        # Create aligner instance
        aligner = ScalesTrialAligner(self.session_dict)
        
        # Run alignment quietly (without all the printing)
        self.aligned_pc_trials = aligner.get_aligned_trials_quiet()
        
        # Store alignment info for thesis values
        self.platform_event_count = len(aligner.platform_events)
        self.total_pc_trials = len(aligner.pc_trials)
        
        # Get transformation parameters from aligner
        # We need to run the full alignment to get these
        aligner.load_scales_data()
        aligner.get_platform_events(min_duration=1.0)
        aligner.load_pc_trials()
        initial_offset = aligner.find_time_offset_minimize()
        initial_results = aligner.align_and_match(initial_offset)
        refined_results = aligner.refine_alignment_with_matches(initial_results)
        final_results = aligner.refine_with_best_matches(refined_results, threshold_ms=80)
        
        # Store transformation parameters
        if 'final_refinement_info' in final_results:
            self.final_scale = final_results['final_refinement_info']['final_scale']
            self.drift_rate = (self.final_scale - 1.0) * 3600 * 1000  # ms/hour
        else:
            self.final_scale = 1.0
            self.drift_rate = 0.0
        
        # Filter for port 4 trials only
        port4_trials = [t for t in self.aligned_pc_trials if t['port'] == 4]
        
        print(f"Found {len(port4_trials)} Port 4 trials in aligned PC logs")
        return port4_trials
    
    def compare_trial_times(self, save_plots=False, output_dir=None):
        """
        Compare DAQ-detected LED times with aligned PC trial times for Port 4.
        
        Args:
            save_plots (bool): Whether to save validation plots
            output_dir (str/Path): Directory to save plots and results
        """
        # Get Port 4 trials from aligned PC logs
        pc_port4_trials = [t for t in self.aligned_pc_trials if t['port'] == 4]
        
        print("\n" + "="*80)
        print("PORT 4 TRIAL TIME COMPARISON")
        print("="*80)
        print(f"\nDAQ LED_4 trials detected: {len(self.daq_trials)}")
        print(f"PC log Port 4 trials: {len(pc_port4_trials)}")
        
        # Match trials by order (assuming they should align sequentially)
        matches_to_compare = min(len(self.daq_trials), len(pc_port4_trials))
        
        if matches_to_compare == 0:
            print("\nNo trials to compare!")
            return {}
        
        print(f"\nComparing {matches_to_compare} trials (matched by order)")
        print("-"*80)
        print(f"{'Trial':<6} {'DAQ LED On':<12} {'PC Aligned':<12} {'Difference':<12} {'PC Original':<12} {'Outcome':<10}")
        print("-"*80)
        
        differences = []
        daq_times = []
        pc_aligned_times = []
        
        for i in range(matches_to_compare):
            daq_trial = self.daq_trials[i]
            pc_trial = pc_port4_trials[i]
            
            # Calculate time difference
            daq_time = daq_trial['led_on_time']
            pc_aligned_time = pc_trial['aligned_time']
            difference = pc_aligned_time - daq_time
            
            differences.append(difference)
            daq_times.append(daq_time)
            pc_aligned_times.append(pc_aligned_time)
            
            # Get outcome
            outcome = pc_trial.get('outcome', 'timeout')
            if outcome is None:
                outcome = 'timeout'
            
            print(f"{i:<6} {daq_time:<12.3f} {pc_aligned_time:<12.3f} {difference:<12.3f} "
                  f"{pc_trial['pc_time']:<12.3f} {outcome:<10}")
        
        # Calculate statistics
        validation_stats = {}
        if differences:
            differences_array = np.array(differences)
            differences_ms = differences_array * 1000  # Convert to milliseconds
            
            print("\n" + "-"*80)
            print("TIMING DIFFERENCE STATISTICS (PC aligned - DAQ)")
            print("-"*80)
            print(f"Mean difference: {np.mean(differences_array):.3f} seconds ({np.mean(differences_ms):.1f} ms)")
            print(f"Std deviation: {np.std(differences_array):.3f} seconds ({np.std(differences_ms):.1f} ms)")
            print(f"Median difference: {np.median(differences_array):.3f} seconds ({np.median(differences_ms):.1f} ms)")
            print(f"Min difference: {np.min(differences_array):.3f} seconds ({np.min(differences_ms):.1f} ms)")
            print(f"Max difference: {np.max(differences_array):.3f} seconds ({np.max(differences_ms):.1f} ms)")
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean(differences_array**2))
            print(f"RMSE: {rmse:.3f} seconds ({rmse*1000:.1f} ms)")
            
            # Count how many are within various thresholds
            print(f"\nTrials within timing thresholds:")
            thresholds = [0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]  # seconds (0.25ms, 0.5ms, 1ms, 2ms, 5ms, 10ms, 20ms)
            threshold_counts = {}
            for threshold in thresholds:
                within = np.sum(np.abs(differences_array) <= threshold)
                percent = (within / len(differences)) * 100
                threshold_counts[threshold] = {'count': within, 'percent': percent}
                print(f"  Within ±{threshold*1000:.2f}ms: {within}/{len(differences)} ({percent:.1f}%)")
            
            # Store validation statistics
            validation_stats = {
                'n_trials': len(differences),
                'mean_diff_ms': np.mean(differences_ms),
                'std_diff_ms': np.std(differences_ms),
                'median_diff_ms': np.median(differences_ms),
                'min_diff_ms': np.min(differences_ms),
                'max_diff_ms': np.max(differences_ms),
                'rmse_ms': rmse * 1000,
                'threshold_counts': threshold_counts,
                'differences': differences,
                'daq_times': daq_times,
                'pc_aligned_times': pc_aligned_times
            }
            
            # Generate plots if requested
            if save_plots and output_dir:
                self.generate_thesis_outputs(validation_stats, output_dir)
        
        # Show any unmatched trials
        if len(self.daq_trials) != len(pc_port4_trials):
            print(f"\n" + "-"*80)
            print("UNMATCHED TRIALS")
            print("-"*80)
            
            if len(self.daq_trials) > len(pc_port4_trials):
                print(f"Extra DAQ trials (no corresponding PC trial):")
                for i in range(len(pc_port4_trials), len(self.daq_trials)):
                    print(f"  Trial {i}: LED on at {self.daq_trials[i]['led_on_time']:.3f}s")
            else:
                print(f"Extra PC trials (no corresponding DAQ LED):")
                for i in range(len(self.daq_trials), len(pc_port4_trials)):
                    trial = pc_port4_trials[i]
                    print(f"  Trial {i}: PC time {trial['pc_time']:.3f}s -> aligned {trial['aligned_time']:.3f}s")
        
        return validation_stats
    
    def generate_thesis_outputs(self, validation_stats, output_dir):
        """
        Generate specific outputs formatted for thesis inclusion.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get R-squared value if not already calculated
        if not hasattr(self, '_r_squared'):
            from scipy.stats import linregress
            daq_times = np.array(validation_stats['daq_times'])
            pc_times = np.array(validation_stats['pc_aligned_times'])
            slope, intercept, r_value, p_value, std_err = linregress(daq_times, pc_times)
            self._r_squared = r_value**2
        
        # Create thesis_values.txt with formatted values for easy insertion
        thesis_values_file = output_path / 'thesis_values.txt'
        with open(thesis_values_file, 'w') as f:
            f.write("THESIS VALUES - Copy these into your thesis text\n")
            f.write("="*60 + "\n\n")
            
            f.write("Platform Event Detection:\n")
            f.write(f"  Platform events detected: {self.platform_event_count}\n")
            f.write(f"  PC trials detected: {self.total_pc_trials}\n")
            f.write(f"  Matched trials: {validation_stats['n_trials']}\n\n")
            
            f.write("Validation Results:\n")
            f.write(f"  Mean error: {validation_stats['mean_diff_ms']:.3f} milliseconds\n")
            f.write(f"  Standard deviation: {validation_stats['std_diff_ms']:.3f} milliseconds\n")
            f.write(f"  RMSE: {validation_stats['rmse_ms']:.3f} milliseconds\n")
            f.write(f"  Median error: {validation_stats['median_diff_ms']:.3f} milliseconds\n\n")
            
            f.write("Accuracy Thresholds:\n")
            within_025ms = validation_stats['threshold_counts'].get(0.00025, {'percent': 0})
            within_05ms = validation_stats['threshold_counts'].get(0.0005, {'percent': 0})
            within_1ms = validation_stats['threshold_counts'].get(0.001, {'percent': 0})
            f.write(f"  Within ±0.25ms: {within_025ms['percent']:.0f}%\n")
            f.write(f"  Within ±0.5ms: {within_05ms['percent']:.0f}%\n")
            f.write(f"  Within ±1ms: {within_1ms['percent']:.0f}%\n\n")
            
            f.write("Clock Drift:\n")
            f.write(f"  Scale factor: {self.final_scale:.8f}\n")
            f.write(f"  Drift rate: {self.drift_rate:.3f} ms/hour\n\n")
            
            f.write(f"R-squared value: {self._r_squared:.6f}\n")
        
        print(f"\nThesis values saved to: {thesis_values_file}")
        
        # Generate the four specific plots mentioned in the thesis
        self._generate_thesis_plots(validation_stats, output_path)
    
    def _generate_thesis_plots(self, stats, output_path):
        """
        Generate the four specific plots mentioned in the thesis text.
        """
        # Set consistent style for thesis
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
        differences_ms = np.array(stats['differences']) * 1000
        
        # 1. Error Distribution Histogram
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        n, bins, patches = ax.hist(differences_ms, bins=30, alpha=0.7, 
                                   edgecolor='black', color='steelblue')
        
        # Add normal distribution overlay
        from scipy import stats as scipy_stats
        mu, sigma = scipy_stats.norm.fit(differences_ms)
        x = np.linspace(differences_ms.min() - 10, differences_ms.max() + 10, 100)
        ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma) * len(differences_ms) * (bins[1] - bins[0]),
                'r-', linewidth=2, label='Normal distribution')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        ax.set_xlabel('Timing Error (ms)', fontsize=12)
        ax.set_ylabel('Number of Trials', fontsize=12)
        ax.set_title('Distribution of Timing Errors', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to show more range
        # Option 1: Fixed range (e.g., -5 to 5 ms)
        ax.set_xlim(-2, 2)
        
        # Option 2: Dynamic range based on data (uncomment to use)
        # max_abs_diff = max(abs(differences_ms.min()), abs(differences_ms.max()))
        # padding = max(2, max_abs_diff * 0.5)  # At least 2ms padding
        # ax.set_xlim(-padding, padding)
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_1_error_distribution.pdf', bbox_inches='tight')
        plt.savefig(output_path / 'figure_1_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Linear Correlation Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        daq_times = np.array(stats['daq_times'])
        pc_times = np.array(stats['pc_aligned_times'])
        
        # Plot points
        ax.scatter(daq_times, pc_times, alpha=0.6, s=40, color='steelblue', 
                   edgecolor='black', linewidth=0.5, label='Trial times')
        
        # Perfect alignment line
        min_time = min(daq_times.min(), pc_times.min())
        max_time = max(daq_times.max(), pc_times.max())
        ax.plot([min_time, max_time], [min_time, max_time], 'r--', 
                linewidth=2, label='Perfect alignment', alpha=0.7)
        
        # Calculate R-squared
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(daq_times, pc_times)
        
        ax.set_xlabel('DAQ LED Time (s)', fontsize=12)
        ax.set_ylabel('Recovered Time (s)', fontsize=12)
        ax.set_title(f'Recovered vs Actual Trial Times (R² = {r_value**2:.6f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_2_linear_correlation.pdf', bbox_inches='tight')
        plt.savefig(output_path / 'figure_2_linear_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store R-squared for thesis values
        self._r_squared = r_value**2
        
        # 3. Temporal Stability Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        session_duration = (daq_times[-1] - daq_times[0]) / 60  # Convert to minutes
        session_times_min = (daq_times - daq_times[0]) / 60
        
        ax.scatter(session_times_min, differences_ms, alpha=0.6, s=40, 
                   color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(stats['mean_diff_ms'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {stats["mean_diff_ms"]:.1f}ms')
        
        # Add trend line to check for drift
        z = np.polyfit(session_times_min, differences_ms, 1)
        p = np.poly1d(z)
        ax.plot(session_times_min, p(session_times_min), "g-", linewidth=2, 
                label=f'Trend: {z[0]:.3f} ms/min')
        
        ax.set_xlabel('Time into Session (minutes)', fontsize=12)
        ax.set_ylabel('Timing Error (ms)', fontsize=12)
        ax.set_title('Temporal Stability of Timing Errors', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to show more range
        # Option 1: Fixed range (e.g., -5 to 5 ms)
        ax.set_ylim(-2, 2)
        
        # Option 2: Dynamic range based on data (uncomment to use)
        # max_abs_diff = max(abs(differences_ms.min()), abs(differences_ms.max()))
        # padding = max(2, max_abs_diff * 0.5)  # At least 2ms padding
        # ax.set_ylim(-padding, padding)
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_3_temporal_stability.pdf', bbox_inches='tight')
        plt.savefig(output_path / 'figure_3_temporal_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Cumulative Accuracy Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        abs_differences_ms = np.abs(differences_ms)
        sorted_diffs = np.sort(abs_differences_ms)
        cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100
        
        ax.plot(sorted_diffs, cumulative, linewidth=3, color='steelblue')
        
        # Add key threshold markers
        key_thresholds = [10, 20, 50]  # milliseconds
        for threshold in key_thresholds:
            if threshold <= sorted_diffs.max():
                idx = np.searchsorted(sorted_diffs, threshold)
                if idx < len(cumulative):
                    percent = cumulative[idx]
                    ax.plot([threshold, threshold], [0, percent], 'k:', linewidth=1.5)
                    ax.plot([0, threshold], [percent, percent], 'k:', linewidth=1.5)
                    ax.annotate(f'{percent:.0f}% within\n±{threshold}ms', 
                               xy=(threshold, percent), xytext=(threshold+5, percent-10),
                               fontsize=10, ha='left')
        
        ax.set_xlabel('Absolute Timing Error (ms)', fontsize=12)
        ax.set_ylabel('Cumulative Percentage of Trials (%)', fontsize=12)
        ax.set_title('Cumulative Distribution of Timing Accuracy', fontsize=14)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(0, max(5, sorted_diffs.max() * 1.1))  # Show at least 5ms range
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_4_cumulative_accuracy.pdf', bbox_inches='tight')
        plt.savefig(output_path / 'figure_4_cumulative_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nThesis plots saved to: {output_path}")
        print("  - figure_1_error_distribution.pdf")
        print("  - figure_2_linear_correlation.pdf")
        print("  - figure_3_temporal_stability.pdf")
        print("  - figure_4_cumulative_accuracy.pdf")
    
    def run_comparison(self, save_plots=False, output_dir=None):
        """
        Run the complete comparison process.
        
        Args:
            save_plots (bool): Whether to save validation plots
            output_dir (str/Path): Directory to save plots and results
        """
        print(f"\nComparing Port 4 trial times for session {self.session_id}")
        print("="*80)
        
        # Load LED data and detect DAQ trials
        print("\nStep 1: Detecting trials from DAQ LED_4 trace...")
        self.load_led4_data()
        self.detect_daq_trials()
        
        # Get aligned PC trials
        print("\nStep 2: Getting aligned PC trial times...")
        self.get_aligned_pc_trials()
        
        # Compare the times
        print("\nStep 3: Comparing trial times...")
        validation_stats = self.compare_trial_times(save_plots=save_plots, output_dir=output_dir)
        
        return {
            'daq_trials': self.daq_trials,
            'pc_trials': [t for t in self.aligned_pc_trials if t['port'] == 4],
            'validation_stats': validation_stats
        }


# Example usage
if __name__ == "__main__":
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    
    print("Loading cohort info...")
    cohort = Cohort_folder('/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics')
    
    session_dict = cohort.get_session("250624_143350_mtao101-3d")
    
    # Create comparison instance and run with validation plots
    comparator = Port4TrialComparison(session_dict)
    
    # Set output directory for validation plots
    output_dir = Path("/lmb/home/srogers/Dev/projects/hex_behav_analysis/hex_behav_analysis/debug/test_validation_plots")
    
    # Run comparison with plots
    results = comparator.run_comparison(save_plots=True, output_dir=output_dir)