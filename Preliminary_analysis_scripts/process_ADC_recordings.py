import os
import json
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import open_ephys.analysis

def print_memory_usage():
    # Get available memory
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GiB

    # Get memory usage of the current process
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / (1024 ** 3)  # Convert to GiB (resident set size)

    print(f"Available Memory: {available_memory:.2f} GiB, Memory Used by Script: {memory_used:.2f} GiB")



class process_ADC_Recordings:
    def __init__(self, dirname, rig=None):
        self.dirname = dirname
        self.rig = rig or 1

        self.extract_ADC_data()
        self.get_DAQ_pulses()
        self.get_camera_pulses()

    def extract_ADC_data(self):
        self.recording = open_ephys.analysis.Session(self.dirname).recordnodes[0].recordings[0].continuous[0]
        self.samples = self.recording.samples   # raw samples, not in microvolts
        self.sample_numbers = self.recording.sample_numbers
        self.timestamps = self.recording.timestamps
        self.metadata = self.recording.metadata

        self.total_sample_number = len(self.sample_numbers)

        self.ADC_channels = {}

        print_memory_usage()

        # Channels to extract based on rig setup
        channels_in_use = ['ADC1', 'ADC2'] if self.rig == 1 else ['ADC5', 'ADC4']

        # Vectorized access to channel data
        channel_indices = [
            i for i, channel_name in enumerate(self.metadata['channel_names'])
            if channel_name in channels_in_use
        ]

        # Process each channel of interest
        for i in channel_indices:
            self.ADC_channels[self.metadata['channel_names'][i]] = self.channel_data(i)

        # Clean up memory
        del self.samples, self.sample_numbers
        gc.collect()

    def channel_data(self, index):
        # Directly get all samples for the given channel index, avoiding loops
        data = np.array(self.recording.get_samples(start_sample_index=0, end_sample_index=self.total_sample_number))[:, index]
        
        # Clean the data with a vectorized operation
        cleaned_data = self.clean_square_wave(data)
        return cleaned_data

    def clean_square_wave(self, data):
        # Use NumPy for efficient processing
        if len(data) < 300000:
            raise Exception("DAQ data too short for processing")
        window_slice = data[300000:400000]  # Slice a window for determining min/max
        max_value = np.max(window_slice)
        min_value = np.min(window_slice)
        mean_value = (max_value + min_value) / 2

        # Vectorize the square wave cleaning process
        normalised_data = np.where(data > mean_value, 1, 0)
        return normalised_data

    def get_DAQ_pulses(self):
        try:
            data = self.ADC_channels["ADC1"] if self.rig == 1 else self.ADC_channels["ADC5"]
        except KeyError:
            print("Channel not found")
            return

        # Find rising edges in a vectorized manner
        data_diff = np.diff(data, prepend=data[0])

        # Convert timestamps to a NumPy array for proper indexing
        timestamps_np = np.array(self.timestamps)

        self.pulses = timestamps_np[np.where(data_diff == 1)]

        # Clean up memory
        del data, timestamps_np
        gc.collect()

    def get_camera_pulses(self):
        try:
            data = self.ADC_channels["ADC2"] if self.rig == 1 else self.ADC_channels["ADC4"]
        except KeyError:
            print("Channel not found")
            return

        # Find falling edges in a vectorized manner
        data_diff = np.diff(data, prepend=data[0])

        # Convert timestamps to a NumPy array for proper indexing
        timestamps_np = np.array(self.timestamps)

        self.camera_pulses = timestamps_np[np.where(data_diff == -1)]

        # Clean up memory
        del data, timestamps_np
        gc.collect()

    def view_ADC_data(self, *channels_to_plot, filtered=False, start=0, end=150000):
        print(f"Total length samples: {len(self.timestamps)}")
        print(f"Showing from time: {self.timestamps[start]} to {self.timestamps[end]}")
        if len(channels_to_plot) == 0:
            channels_to_plot = self.ADC_channels.keys()

        print(f"Showing channels: {channels_to_plot}, start: {start}, end: {end}")

        if end is None:
            end = len(self.timestamps)

        fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(15, 10))
        if len(channels_to_plot) == 1:
            axs = [axs]

        for i, channel in enumerate(channels_to_plot):
            data = self.ADC_channels[channel][start:end]
            axs[i].plot(self.timestamps[start:end], data, linewidth=0.5)
            axs[i].set_title(channel, loc='left', fontsize=8)
            axs[i].set_ylim([-5, 5])
            axs[i].set_xlim([self.timestamps[start:end][0], self.timestamps[start:end][-1]])

        plt.subplots_adjust(hspace=0.65, left=0.05, right=0.97, top=0.95, bottom=0.05)
        plt.show()

        # Clean up after plotting
        gc.collect()


if __name__ == "__main__":
    # test = process_ADC_Recordings(r"E:\Test_output\240906_001430\240906_001430_OEAB_recording")
    test = process_ADC_Recordings(r"E:\Test_output\240906_024544\240906_024544_OEAB_recording")
    
    test.view_ADC_data("ADC2")
