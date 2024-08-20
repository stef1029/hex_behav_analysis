import os
import json
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import open_ephys.analysis

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

        available_memory = psutil.virtual_memory().available
        print(f"Available Memory: {available_memory / (1024 ** 3)} GiB")

        channels_in_use = ['ADC1', 'ADC2'] if self.rig == 1 else ['ADC5', 'ADC4']

        for i in range(self.metadata['num_channels']):
            channel_name = self.metadata['channel_names'][i]
            if channel_name in channels_in_use:
                self.ADC_channels[channel_name] = self.channel_data(i)
        
        # Delete samples and sample_numbers to free memory if no longer needed
        del self.samples, self.sample_numbers
        gc.collect()  # Force garbage collection

    def channel_data(self, index):
        data = []
        for row in self.recording.get_samples(start_sample_index=0, end_sample_index=self.total_sample_number):
            data.append(row[index])
        
        # Clean the data and release memory as soon as possible
        cleaned_data = self.clean_square_wave(data)
        del data
        gc.collect()  # Force garbage collection
        return cleaned_data
    
    def clean_square_wave(self, data):
        max_value = max(data[1000:10000])
        min_value = min(data[1000:10000])
        mean_value = (max_value + min_value) / 2

        normalised_data = np.where(np.array(data) > mean_value, 1, 0).tolist()

        return normalised_data

    def get_DAQ_pulses(self):
        try:
            data = self.ADC_channels["ADC1"] if self.rig == 1 else self.ADC_channels["ADC5"]
        except KeyError:
            print("Channel not found")
            return

        self.pulses = []
        for i, datapoint in enumerate(data):
            if datapoint == 1 and data[i-1] == 0:
                self.pulses.append(self.timestamps[i])

        # Delete data to free memory
        del data
        gc.collect()  # Force garbage collection

    def get_camera_pulses(self):
        try:
            data = self.ADC_channels["ADC2"] if self.rig == 1 else self.ADC_channels["ADC4"]
        except KeyError:
            print("Channel not found")
            return

        self.camera_pulses = []
        for i, datapoint in enumerate(data):
            if datapoint == 0 and data[i-1] == 1:
                self.camera_pulses.append(self.timestamps[i])

        # Delete data to free memory
        del data
        gc.collect()  # Force garbage collection

    def import_arduino_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        messages = data["messages"]
        for i, message in enumerate(messages):
            try:
                message.append(self.pulses[i])
            except IndexError:
                print(f"IndexError at message {i}")
                break
        filename = f"{self.dirname.name}_arduino_data.json"
        with open(filename, 'w') as f:
            json.dump(messages, f)

        # Clean up to release memory
        del data, messages
        gc.collect()  # Force garbage collection

        return messages

    def view_ADC_data(self, *channels_to_plot, filtered=False, start=0, end=None):
        if len(channels_to_plot) == 0:
            channels_to_plot = self.ADC_channels.keys()

        print(f"Showing channels: {channels_to_plot}, start: {start}, end: {end}")

        if end is None:
            end = len(self.timestamps)

        fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(15, 10))

        for i, channel in enumerate(channels_to_plot):
            data = self.ADC_channels[channel][start:end]

            axs[i].scatter(self.timestamps[start:end], data, s=0.1)
            axs[i].set_title(channel, loc='left', fontsize=8)

        plt.setp(axs, ylim=[-5, 5], xlim=[self.timestamps[0], self.timestamps[-1]])
        plt.subplots_adjust(hspace=0.65, left=0.05, right=0.97, top=0.95, bottom=0.05)
        plt.show()

        # Clean up after plotting
        del data
        gc.collect()  # Force garbage collection

if __name__ == "__main__":
    test = process_ADC_Recordings(r"/cephfs2/srogers/March_training/240327_160116/2024-03-27_16-01-36")
    test.view_ADC_data("ADC1")