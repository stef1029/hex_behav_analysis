import json
import matplotlib.pyplot as plt
import h5py
import numpy as np
from Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import
import argparse
import os

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

    def plot_channel_data(self, channel, start, end):
        """
        Plot data for a specified channel between start and end indices.
        :param channel: The channel to be plotted
        :param start: Start index
        :param end: End index
        """
        # Check if the channel exists
        if channel not in self.channel_data:
            print(f"Channel '{channel}' not found. Available channels: {list(self.channel_data.keys())}")
            return
        
        # Ensure start and end are within bounds
        start = max(0, start)
        end = min(len(self.channel_data[channel]), end)

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
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="View and plot data from a DAQ file.")
    parser.add_argument('--file', type=str, required=True, help='Path to the DAQ file (either JSON or HDF5)')
    parser.add_argument('--channel', type=str, required=True, help='Channel to plot')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=1000, help='End index')
    args = parser.parse_args()

    # Initialize DAQViewer with the provided file
    viewer = DAQViewer(args.file)
    
    # Plot the data for the specified channel and range
    viewer.plot_channel_data(args.channel, args.start, args.end)

if __name__ == "__main__":
    main()
