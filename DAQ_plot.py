import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sig

class DAQ_plot:
    def __init__(self, DAQ_data, directory, debug = False, trials = None):
        self.DAQ_data = DAQ_data
        self.output_path = directory
        self.debug = debug
        self.trials = trials

        self.plot_all(self.DAQ_data)

    def filter_data(self, original_data):
        # Filter requirements.
        T = 800.0         # Sample Period
        fs = 1000.0       # sample rate, Hz
        cutoff = 10  # desired cutoff frequency of the filter, Hz ,  
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = sig.butter(order, normal_cutoff, btype='lowpass', analog=False)       
        filtered_data = sig.filtfilt(b, a, original_data)

        return filtered_data

    def plot_all(self, DAQ_data, start = 0, end = None):

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

        if end == None:
            end = len(self.DAQ_data["timestamps"])-1
            
        for i, channel in enumerate(channel_order):

            # get channel data and convert to floats:
            data = [float(datapoint) for datapoint in self.DAQ_data[channel]]
            float_timestamps = [float(timestamp) for timestamp in self.DAQ_data["timestamps"]]       # that bug i couldn't figure out for ages was because it was plotting strings, not floats.

            # if SPOT in channel name:
            if "SPOT" in channel:
                # filter data:
                data = self.filter_data(data)

            # axs[i].scatter(float_timestamps[start: end], data[start: end], s = 0.05, marker = '.')
            axs[i].plot(float_timestamps[start: end], data[start: end], linewidth=0.5)
            
            # set title location to left of plot using x y coordinates, so they are actually to the left and not above the graph at all:
            axs[i].set_title(channel, loc = 'left', fontsize = 8, x = -0.05, y = -0.3)

            axs[i].set_ylim(bottom = -0.1, top = 1.1)

            axs[i].set_yticks([0, 1])

        scales_weights = self.DAQ_data["scales_data"]["weights"]
        scales_timestamps = self.DAQ_data["scales_data"]["timestamps"]
        threshold = self.DAQ_data["scales_data"]["mouse_weight_threshold"]

        axs[-1].plot(scales_timestamps, scales_weights, linewidth=0.5)
        
        axs[-1].set_title("Scales", loc = 'left', fontsize = 8, x = -0.05, y = -0.3)

        max_weight = max(scales_weights)
        
        # If your threshold value is greater than the max_weight, you won't see the line. 
        # Make sure y-limit includes the threshold. 
        axs[-1].set_ylim([0, max_weight if max_weight > threshold else threshold])

        axs[-1].set_yticks([0, max_weight])

        # Highlight areas where weight is above threshold
        start_above_threshold = None
        for i, (time, weight) in enumerate(zip(scales_timestamps, scales_weights)):
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

        axs[-1].plot([scales_timestamps[0], scales_timestamps[-1]], [threshold, threshold], color='red', linewidth=0.5, zorder=10)

        if self.debug == True and self.trials != None:
            print("Plotting trials")
            # Highlight trials
            # use axs on led channels to plot trials
            for i, led in enumerate(["LED_1", "LED_2", "LED_3", "LED_4", "LED_5", "LED_6"]):
                for trial in self.trials:
                    # convert indices to timestamps:
                    trial_start = float_timestamps[int(trial["start"])]
                    trial_end = float_timestamps[int(trial["end"])]
                    start_frame = float(trial["video_frames"][0]["time"])
                    end_frame = float(trial["video_frames"][-1]["time"])
                    print(f"Start: {trial_start}, End: {trial_end}, Start frame: {start_frame}, End frame: {end_frame}")
                    axs[i+12].axvspan(trial_start, trial_end, facecolor='lightblue', alpha=1, ymin=0.5, ymax=1)
                    axs[i+12].axvspan(start_frame, end_frame, facecolor='lightgreen', alpha=1, ymin=0, ymax=0.5)
            
        
            
        plt.setp(axs, xlim=[DAQ_data["timestamps"][start], DAQ_data["timestamps"][end]])

        # x ticks should be timestamps, every 5 minutes, but only on bottom plot. data currently in seconds, so convert to minutes:
        # axs[-1].set_xticklabels([round(timestamp / 60, 1) for timestamp in DAQ_data["timestamps"][::300]])
        # set spacing between subplots:
        plt.subplots_adjust(hspace=0.65, left = 0.05, right = 0.97, top = 0.95, bottom = 0.05)


        # save figure at high resolution:
        filename = self.output_path / f"{self.output_path.stem}.png"
        if self.debug:
            filename = self.output_path / f"{self.output_path.stem}_debug.png"
        plt.savefig(filename, dpi = 1200)
