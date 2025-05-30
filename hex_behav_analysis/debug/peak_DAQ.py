import json
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import os

from hex_behav_analysis.Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import

class DAQViewer:
    def __init__(self, daq_file_path):
        # Determine file type based on extension
        file_extension = os.path.splitext(daq_file_path)[1].lower()

        if file_extension == '.json':
            self.load_from_json(daq_file_path)
        elif file_extension == '.h5':
            self.load_from_hdf5(daq_file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .json or .h5 file.")

    def load_from_json(self, daq_file_path):
        """Load channel data from a JSON file."""
        self.daq_importer = Arduino_DAQ_Import(daq_file_path)
        self.channel_data = self.daq_importer.channel_data
        self.message_ids = self.daq_importer.channel_data["message_ids"]

    def load_from_hdf5(self, daq_file_path):
        """Load channel data from an HDF5 file."""
        with h5py.File(daq_file_path, 'r') as h5f:
            # Load datasets from the HDF5 file
            self.message_ids = h5f['message_ids'][:]
            self.timestamps = h5f['timestamps'][:]
            self.channel_data = {}

            # Load all channels from the 'channel_data' group
            channel_group = h5f['channel_data']
            for channel in channel_group:
                self.channel_data[channel] = channel_group[channel][:]

    def plot_channel_data(self, channel, start, end, save=False, show=True):
        """
        Plot data for a specified channel between start and end indices.
        Optionally save the plot and/or display it.
        
        :param channel: The channel to be plotted
        :param start: Start index
        :param end: End index
        :param save: Whether to save the plot to 'temp_output'
        :param show: Whether to display the plot
        """
        # Check if the channel exists
        if channel not in self.channel_data:
            print(f"Channel '{channel}' not found. Available channels: {list(self.channel_data.keys())}")
            return
        
        # Ensure start and end are within bounds
        start = max(0, start)
        if end != None:
            end = min(len(self.channel_data[channel]), end)
        else:
            end = len(self.channel_data[channel])

        if start >= end:
            print("Invalid start and end indices. Please ensure start < end.")
            return

        # Extract the data for the given channel and range
        data_to_plot = self.channel_data[channel][start:end]
        time_axis = range(start, end)

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, data_to_plot, label=channel)
        plt.title(f"Data for Channel '{channel}' from {start} to {end}")
        plt.xlabel("Time (sample index)")
        plt.ylabel("Signal")
        plt.legend()
        plt.grid(True)
        
        # Save the plot if required
        if save:
            output_dir = "temp_output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{channel}_plot_{start}_{end}.png")
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        # Show the plot if required
        if show:
            plt.show()
        else:
            plt.close()

def main():

    show = True
    save = False

    # file = r"/cephfs2/dwelch/Behaviour/November_cohort/250131_112302/250131_112303_wtjp273-3f/250131_112303_wtjp273-3f-ArduinoDAQ.h5"
    file = r"Z:\Behaviour code\2409_September_cohort\DATA_ArduinoDAQ\241017_151131\241017_151132_mtao89-1e\241017_151132_mtao89-1e-ArduinoDAQ.h5"
    channel = 'CAMERA'
    start = 0 
    end = 10000
    # Initialize DAQViewer with the provided file
    viewer = DAQViewer(file)
    
    # Plot the data for the specified channel and range
    viewer.plot_channel_data(channel, start, end, save = save, show = show)

if __name__ == "__main__":
    main()
