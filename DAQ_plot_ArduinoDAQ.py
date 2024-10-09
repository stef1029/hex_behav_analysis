import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sig
import numpy as np
import h5py

class DAQ_plot:
    def __init__(self, DAQ_h5_path, directory, scales_data, debug=False, trials=None):
        self.DAQ_h5_path = DAQ_h5_path
        self.output_path = directory
        self.debug = debug
        self.trials = trials
        self.scales_data = scales_data

        # Read data from the HDF5 file
        self.read_h5_data()

        # Generate the plots
        self.plot_all()

    def read_h5_data(self):
        with h5py.File(self.DAQ_h5_path, 'r') as h5f:
            # Read the timestamps
            self.timestamps = np.array(h5f['timestamps'], dtype=np.float64)

            # Read the channel data
            self.channel_data = {}
            for channel_name in h5f['channel_data']:
                self.channel_data[channel_name] = np.array(h5f['channel_data'][channel_name], dtype=np.int8)

    def filter_data(self, original_data):
        # Filter requirements.
        T = 800.0         # Total time duration (not used in filtering)
        fs = 1000.0       # Sample rate, Hz
        cutoff = 10       # Desired cutoff frequency of the filter, Hz
        nyq = 0.5 * fs    # Nyquist Frequency
        order = 2         # Order of the filter

        normal_cutoff = cutoff / nyq
        b, a = sig.butter(order, normal_cutoff, btype='lowpass', analog=False)
        filtered_data = sig.filtfilt(b, a, original_data)

        return filtered_data

    def plot_all(self, start=0, end=None):
        channel_order = (
            "SPOT1", "SPOT2", "SPOT3", "SPOT4", "SPOT5", "SPOT6", 
            "SENSOR1", "LED_1", "VALVE1", 
            "SENSOR2", "LED_2", "VALVE2", 
            "SENSOR3", "LED_3", "VALVE3", 
            "SENSOR4", "LED_4", "VALVE4", 
            "SENSOR5", "LED_5", "VALVE5", 
            "SENSOR6", "LED_6", "VALVE6", 
            "BUZZER1", "BUZZER2", "BUZZER3", "BUZZER4", "BUZZER5", "BUZZER6", 
            "GO_CUE", "NOGO_CUE"
        )

        scales_axes = 1

        fig = plt.figure(figsize=(18, 10))

        gs = gridspec.GridSpec(len(channel_order) + scales_axes, 1, height_ratios=[1]*len(channel_order) + [3])
        axs = [fig.add_subplot(gs[i]) for i in range(len(channel_order) + scales_axes)]

        if end is None:
            end = len(self.timestamps) - 1

        float_timestamps = self.timestamps.astype(float)

        for i, channel in enumerate(channel_order):
            if channel in self.channel_data:
                data = self.channel_data[channel].astype(float)
            else:
                # If the channel is not found, fill with zeros
                data = np.zeros_like(float_timestamps)
                print(f"Channel {channel} not found in data. Filling with zeros.")

            # Apply filtering if channel is a SPOT channel
            if "SPOT" in channel:
                data = self.filter_data(data)

            axs[i].plot(float_timestamps[start:end], data[start:end], linewidth=0.5)

            # Set title location to the left of the plot
            axs[i].set_title(channel, loc='left', fontsize=8, x=-0.05, y=-0.3)

            axs[i].set_ylim(bottom=-0.1, top=1.1)
            axs[i].set_yticks([0, 1])

        # Plot scales data
        scales_weights = np.array(self.scales_data["weights"], dtype=np.float64)
        scales_timestamps = np.array(self.scales_data["timestamps"], dtype=np.float64)
        threshold = self.scales_data["mouse_weight_threshold"]

        axs[-1].plot(scales_timestamps, scales_weights, linewidth=0.5)

        axs[-1].set_title("Scales", loc='left', fontsize=8, x=-0.05, y=-0.3)

        max_weight = max(scales_weights)

        # Ensure y-limit includes the threshold
        axs[-1].set_ylim([0, max(max_weight, threshold)])
        axs[-1].set_yticks([0, max_weight])

        # Highlight areas where weight is above threshold
        start_above_threshold = None
        for time, weight in zip(scales_timestamps, scales_weights):
            if weight > threshold and start_above_threshold is None:
                # Start of a new region above threshold
                start_above_threshold = time
            elif weight <= threshold and start_above_threshold is not None:
                # End of the region above threshold, draw highlight
                axs[-1].axvspan(start_above_threshold, time, facecolor='lightgreen', alpha=1)
                start_above_threshold = None
        # If data ends while still above threshold
        if start_above_threshold is not None:
            axs[-1].axvspan(start_above_threshold, scales_timestamps[-1], facecolor='lightgreen', alpha=0.5)

        # Plot the threshold line
        axs[-1].plot([scales_timestamps[0], scales_timestamps[-1]], [threshold, threshold],
                     color='red', linewidth=0.5, zorder=10)

        if self.debug and self.trials is not None:
            print("Plotting trials")
            # Highlight trials on LED channels
            led_indices = [channel_order.index(led) for led in ["LED_1", "LED_2", "LED_3", "LED_4", "LED_5", "LED_6"]]
            for idx, led in zip(led_indices, ["LED_1", "LED_2", "LED_3", "LED_4", "LED_5", "LED_6"]):
                for trial in self.trials:
                    # Convert indices to timestamps
                    trial_start = float_timestamps[int(trial["start"])]
                    trial_end = float_timestamps[int(trial["end"])]
                    start_frame = float(trial["video_frames"][0]["time"])
                    end_frame = float(trial["video_frames"][-1]["time"])
                    print(f"Start: {trial_start}, End: {trial_end}, Start frame: {start_frame}, End frame: {end_frame}")
                    axs[idx].axvspan(trial_start, trial_end, facecolor='lightblue', alpha=1, ymin=0.5, ymax=1)
                    axs[idx].axvspan(start_frame, end_frame, facecolor='lightgreen', alpha=1, ymin=0, ymax=0.5)

        # Set x-limits for all axes
        plt.setp(axs, xlim=[self.timestamps[start], self.timestamps[end]])

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.65, left=0.05, right=0.97, top=0.95, bottom=0.05)

        # Save figure at high resolution
        filename = self.output_path / f"{self.output_path.stem}.png"
        if self.debug:
            filename = self.output_path / f"{self.output_path.stem}_debug.png"
        plt.savefig(filename, dpi=1200)
        plt.close(fig)  # Close the figure to free memory
