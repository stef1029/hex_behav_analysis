import json
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import os
import socket

from hex_behav_analysis.Preliminary_analysis_scripts.Full_arduinoDAQ_import import Arduino_DAQ_Import


class DAQViewer:
    """
    A class for loading and visualising data from Arduino DAQ files.
    
    Supports both JSON and HDF5 file formats and allows plotting of single
    or multiple channels with customisable time ranges.
    """
    
    def __init__(self, daq_file_path):
        """
        Initialise the DAQViewer with a DAQ file.
        
        Args:
            daq_file_path (str): Path to the DAQ file (.json or .h5)
        
        Raises:
            ValueError: If the file format is not supported
        """
        # Determine file type based on extension
        file_extension = os.path.splitext(daq_file_path)[1].lower()

        if file_extension == '.json':
            self.load_from_json(daq_file_path)
        elif file_extension == '.h5':
            self.load_from_hdf5(daq_file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .json or .h5 file.")

    def load_from_json(self, daq_file_path):
        """
        Load channel data from a JSON file.
        
        Args:
            daq_file_path (str): Path to the JSON file
        """
        self.daq_importer = Arduino_DAQ_Import(daq_file_path)
        self.channel_data = self.daq_importer.channel_data
        self.message_ids = self.daq_importer.channel_data["message_ids"]
        # For consistency with HDF5, try to extract timestamps if available
        self.timestamps = self.daq_importer.channel_data.get("timestamps", None)

    def load_from_hdf5(self, daq_file_path):
        """
        Load channel data from an HDF5 file.
        
        Args:
            daq_file_path (str): Path to the HDF5 file
        """
        with h5py.File(daq_file_path, 'r') as h5f:
            # Load datasets from the HDF5 file
            self.message_ids = h5f['message_ids'][:]
            self.timestamps = h5f['timestamps'][:]
            self.channel_data = {}

            # Load all channels from the 'channel_data' group
            channel_group = h5f['channel_data']
            for channel in channel_group:
                self.channel_data[channel] = channel_group[channel][:]

    def plot_channel_data(self, channels, start=0, end=None, save=False, show=True, use_timestamps=True, web_display=False):
        """
        Plot data for specified channel(s) between start and end indices.
        
        Args:
            channels (str or list): Single channel name or list of channel names to plot
            start (int): Start index for the plot (default: 0)
            end (int or None): End index for the plot (default: None, plots to end of data)
            save (bool): Whether to save the plot to 'temp_output' directory
            show (bool): Whether to display the plot
            use_timestamps (bool): Whether to use timestamps (if available) or sample indices
            web_display (bool): Whether to display the plot in a web browser on localhost
        
        Returns:
            None
        """
        # Convert single channel to list for uniform handling
        if isinstance(channels, str):
            channels = [channels]
        
        # Validate all channels exist
        invalid_channels = [ch for ch in channels if ch not in self.channel_data]
        if invalid_channels:
            print(f"Channels not found: {invalid_channels}")
            print(f"Available channels: {list(self.channel_data.keys())}")
            return
        
        # Determine data length from first channel
        data_length = len(self.channel_data[channels[0]])
        
        # Ensure start and end are within bounds
        start = max(0, start)
        end = min(data_length, end) if end is not None else data_length

        if start >= end:
            print("Invalid start and end indices. Please ensure start < end.")
            return

        # Configure matplotlib backend for web display if requested
        if web_display:
            matplotlib.use('webagg')
            # Prevent automatic browser opening
            import matplotlib.pyplot as plt
            plt.rcParams['webagg.open_in_browser'] = False
            
        # Create figure with subplots
        num_channels = len(channels)
        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels))
        
        # Handle single subplot case
        if num_channels == 1:
            axes = [axes]
        
        # Determine x-axis values
        if use_timestamps and hasattr(self, 'timestamps') and self.timestamps is not None:
            x_values = self.timestamps[start:end]
            x_label = "Time (s)"
        else:
            x_values = range(start, end)
            x_label = "Sample index"
        
        # Plot each channel
        for idx, (ax, channel) in enumerate(zip(axes, channels)):
            # Extract the data for the given channel and range
            data_to_plot = self.channel_data[channel][start:end]
            
            # Plot the data
            ax.plot(x_values, data_to_plot, label=channel)
            ax.set_ylabel('Signal')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Only show x-label for bottom subplot
            if idx == num_channels - 1:
                ax.set_xlabel(x_label)
            else:
                ax.set_xticklabels([])
        
        # Add title
        if num_channels == 1:
            plt.suptitle(f"Data for Channel '{channels[0]}' from {start} to {end}")
        else:
            plt.suptitle(f"Multi-Channel Data from {start} to {end}")
        
        plt.tight_layout()
        
        # Save the plot if required
        if save:
            output_dir = "temp_output"
            os.makedirs(output_dir, exist_ok=True)
            channels_str = "_".join(channels) if len(channels) <= 3 else f"{channels[0]}_and_{len(channels)-1}_more"
            output_path = os.path.join(output_dir, f"{channels_str}_plot_{start}_{end}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")

        # Show the plot if required
        if show:
            if web_display:
                # Get local IP address
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                port = 8988  # Default WebAgg port
                
                print("\n" + "="*50)
                print("Web server started!")
                print(f"Access the plot at: http://{local_ip}:{port}")
                print(f"Or locally at: http://localhost:{port}")
                print("Press Ctrl+C in the terminal to stop the server")
                print("="*50 + "\n")
            plt.show()
        else:
            plt.close()


def main():
    """
    Main function to demonstrate the usage of DAQViewer with multiple channels.
    """
    # Configuration parameters
    show_plot = True
    save_plot = False
    web_display = True  # Set to True to display in web browser
    
    # File path to the DAQ data
    # daq_file_path = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment/250603_131534/250603_131545_mtao108-3e/250603_131545_mtao108-3e-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/2504_pitx_ephys_cohort/250521_150121/250521_150126_mtaq14-1j/250521_150126_mtaq14-1j-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment/250624_143340/250624_143350_mtao101-3c/250624_143350_mtao101-3c-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/250623_113539/250623_113554_mtao101-3c/250623_113554_mtao101-3c-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/2504_pitx_ephys_cohort/250416_182113/250416_182121_mtaq14-1j/250416_182121_mtaq14-1j-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/2504_pitx_ephys_cohort/250418_143718/250418_143823_mtaq13-3a/250418_143823_mtaq13-3a-ArduinoDAQ.h5"
    # daq_file_path = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment/250610_122933/250610_122940_mtao106-3a/250610_122940_mtao106-3a-ArduinoDAQ.h5"
    daq_file_path = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment/250527_121902/250527_121911_mtao106-3a/250527_121911_mtao106-3a-ArduinoDAQ.h5"
    # "Z:\Behaviour\2504_pitx_ephys_cohort\250521_150121\250521_150126_mtaq14-1j\250521_150126_mtaq14-1j-ArduinoDAQ.h5"
    # Channels to plot - can be a single channel or a list of channels
    channels_to_plot = ['SCALES']  # Modify as needed
    # channels_to_plot = ['CAMERA', 'SENSOR1', 'LED_1', 'VALVE1', 'SENSOR2', 'LED_2', 'VALVE2',
    #                     'SENSOR3', 'LED_3', 'VALVE3', 'SENSOR4', 'LED_4', 'VALVE4',
    #                     'SENSOR5', 'LED_5', 'VALVE5', 'SENSOR6', 'LED_6', 'VALVE6',
    #                     # 'BUZZER1', 'BUZZER2', 'BUZZER3', 'BUZZER4', 'BUZZER5', 'BUZZER6', 
    #                     'GO_CUE', 'NOGO_CUE']
    # channels_to_plot = [
    #     # "SPOT1", "SPOT2", "SPOT3", "SPOT4", "SPOT5", "SPOT6", 
    #         "SENSOR1", "LED_1", "VALVE1", 
    #         "SENSOR2", "LED_2", "VALVE2", 
    #         "SENSOR3", "LED_3", "VALVE3", 
    #         "SENSOR4", "LED_4", "VALVE4", 
    #         "SENSOR5", "LED_5", "VALVE5", 
    #         "SENSOR6", "LED_6", "VALVE6", 
    #         # "BUZZER1", "BUZZER2", "BUZZER3", "BUZZER4", "BUZZER5", "BUZZER6", 
    #         "GO_CUE", "NOGO_CUE"]
    # Time range for plotting
    start_index = 0
    end_index = None
    
    # Initialise DAQViewer with the provided file
    viewer = DAQViewer(daq_file_path)
    
    # Plot the data for the specified channel(s) and range
    viewer.plot_channel_data(
        channels=channels_to_plot,
        start=start_index,
        end=end_index,
        save=save_plot,
        show=show_plot,
        web_display=web_display
    )


if __name__ == "__main__":
    main()