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

    def calculate_pulse_frequency(self, data, timestamps=None, threshold=0.5, window_size_seconds=1.0, sample_rate=None):
        """
        Calculate the frequency of pulses over time using a sliding window.
        
        Args:
            data (np.array): The signal data
            timestamps (np.array or None): Time values for each sample. If None, uses sample indices
            threshold (float): Threshold value for pulse detection (default: 0.5)
            window_size_seconds (float): Size of the sliding window in seconds (default: 1.0)
            sample_rate (float or None): Sample rate in Hz. Required if timestamps is None
        
        Returns:
            tuple: (time_points, frequencies) - Arrays of time points and corresponding frequencies
        """
        # Detect pulse edges (rising edges where signal crosses threshold)
        above_threshold = data >= threshold
        pulse_edges = np.diff(above_threshold.astype(int))
        pulse_indices = np.where(pulse_edges == 1)[0] + 1  # +1 because diff reduces length by 1
        
        if len(pulse_indices) == 0:
            # No pulses detected
            if timestamps is not None:
                return np.array([timestamps[0], timestamps[-1]]), np.array([0, 0])
            else:
                return np.array([0, len(data) - 1]), np.array([0, 0])
        
        # Determine time values for pulses
        if timestamps is not None:
            pulse_times = timestamps[pulse_indices]
            total_duration = timestamps[-1] - timestamps[0]
            time_step = np.median(np.diff(timestamps))  # Estimate time step from timestamps
        else:
            if sample_rate is None:
                # Assume a default sample rate if not provided
                sample_rate = 1000.0  # 1 kHz default
                print(f"Warning: No timestamps or sample rate provided. Assuming {sample_rate} Hz.")
            pulse_times = pulse_indices / sample_rate
            total_duration = len(data) / sample_rate
            time_step = 1.0 / sample_rate
        
        # Calculate window size in samples
        window_samples = int(window_size_seconds / time_step)
        
        # Create time points for frequency calculation
        if timestamps is not None:
            time_points = np.arange(timestamps[0], timestamps[-1], window_size_seconds / 2)
        else:
            time_points = np.arange(0, total_duration, window_size_seconds / 2)
        
        # Calculate frequency at each time point
        frequencies = []
        half_window = window_size_seconds / 2
        
        for t in time_points:
            # Count pulses within the window
            pulses_in_window = np.sum((pulse_times >= t - half_window) & (pulse_times < t + half_window))
            # Convert to frequency (pulses per second)
            frequency = pulses_in_window / window_size_seconds
            frequencies.append(frequency)
        
        return time_points, np.array(frequencies)

    def plot_channel_data(self, channels, start=0, end=None, save=False, show=True, use_timestamps=True, 
                         web_display=False, plot_frequency=False, frequency_threshold=0.5, 
                         frequency_window_size=1.0):
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
            plot_frequency (bool): Whether to plot pulse frequency below each channel (default: False)
            frequency_threshold (float): Threshold for pulse detection (default: 0.5)
            frequency_window_size (float): Window size in seconds for frequency calculation (default: 1.0)
        
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
            
        # Determine number of subplots needed
        num_channels = len(channels)
        subplots_per_channel = 2 if plot_frequency else 1
        total_subplots = num_channels * subplots_per_channel
        
        # Create figure with subplots
        fig, axes = plt.subplots(total_subplots, 1, figsize=(12, 3 * total_subplots))
        
        # Handle single subplot case
        if total_subplots == 1:
            axes = [axes]
        
        # Determine x-axis values
        if use_timestamps and hasattr(self, 'timestamps') and self.timestamps is not None:
            x_values = self.timestamps[start:end]
            x_label = "Time (s)"
        else:
            x_values = range(start, end)
            x_label = "Sample index"
        
        # Plot each channel
        for channel_idx, channel in enumerate(channels):
            # Calculate subplot indices
            signal_ax_idx = channel_idx * subplots_per_channel
            
            # Extract the data for the given channel and range
            data_to_plot = self.channel_data[channel][start:end]
            
            # Plot the signal data
            axes[signal_ax_idx].plot(x_values, data_to_plot, label=channel)
            axes[signal_ax_idx].set_ylabel('Signal')
            axes[signal_ax_idx].legend(loc='upper right')
            axes[signal_ax_idx].grid(True, alpha=0.3)
            
            # Add threshold line if plotting frequency
            if plot_frequency:
                axes[signal_ax_idx].axhline(y=frequency_threshold, color='r', linestyle='--', 
                                          alpha=0.5, label=f'Threshold ({frequency_threshold})')
            
            # Hide x-axis labels except for the last subplot
            if not (channel_idx == num_channels - 1 and not plot_frequency):
                axes[signal_ax_idx].set_xticklabels([])
            
            # Plot frequency if requested
            if plot_frequency:
                freq_ax_idx = signal_ax_idx + 1
                
                # Calculate frequency over time
                if use_timestamps and hasattr(self, 'timestamps') and self.timestamps is not None:
                    freq_time, frequencies = self.calculate_pulse_frequency(
                        data_to_plot, 
                        timestamps=self.timestamps[start:end],
                        threshold=frequency_threshold,
                        window_size_seconds=frequency_window_size
                    )
                else:
                    # Estimate sample rate from timestamps if available
                    sample_rate = None
                    if hasattr(self, 'timestamps') and self.timestamps is not None:
                        time_diffs = np.diff(self.timestamps)
                        sample_rate = 1.0 / np.median(time_diffs)
                    
                    freq_time, frequencies = self.calculate_pulse_frequency(
                        data_to_plot,
                        timestamps=None,
                        threshold=frequency_threshold,
                        window_size_seconds=frequency_window_size,
                        sample_rate=sample_rate
                    )
                    # Adjust freq_time to match the start index
                    freq_time = freq_time + start
                
                # Plot frequency
                axes[freq_ax_idx].plot(freq_time, frequencies, color='green', linewidth=2)
                axes[freq_ax_idx].set_ylabel('Frequency (Hz)')
                axes[freq_ax_idx].grid(True, alpha=0.3)
                axes[freq_ax_idx].set_ylim(bottom=0)  # Frequency cannot be negative
                
                # Only show x-label for the last subplot
                if channel_idx == num_channels - 1:
                    axes[freq_ax_idx].set_xlabel(x_label)
                else:
                    axes[freq_ax_idx].set_xticklabels([])
        
        # Add title
        if num_channels == 1:
            title = f"Data for Channel '{channels[0]}' from {start} to {end}"
            if plot_frequency:
                title += f"\n(Frequency calculated with {frequency_window_size}s window)"
        else:
            title = f"Multi-Channel Data from {start} to {end}"
            if plot_frequency:
                title += f"\n(Frequency calculated with {frequency_window_size}s window)"
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the plot if required
        if save:
            output_dir = "temp_output"
            os.makedirs(output_dir, exist_ok=True)
            channels_str = "_".join(channels) if len(channels) <= 3 else f"{channels[0]}_and_{len(channels)-1}_more"
            suffix = "_with_freq" if plot_frequency else ""
            output_path = os.path.join(output_dir, f"{channels_str}_plot_{start}_{end}{suffix}.png")
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
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    # Configuration parameters
    show_plot = True
    save_plot = False
    web_display = True  # Set to True to display in web browser
    
    # Frequency analysis parameters
    plot_frequency = True  # Set to True to plot frequency analysis
    frequency_threshold = 0.5  # Threshold for pulse detection
    frequency_window_size = 1.0  # Window size in seconds for frequency calculation

    cohort_dir = "/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment"
    session_id = "250521_133747_mtao106-3b"  # Example session ID

    cohort = Cohort_folder(cohort_dir, use_existing_cohort_info=True, OEAB_legacy=False)
    daq_file_path = cohort.get_session(session_id)['raw_data']['arduino_DAQ_h5']
    
    # Channels to plot - can be a single channel or a list of channels
    channels_to_plot = ['CAMERA']  # Modify as needed
    
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
        web_display=web_display,
        plot_frequency=plot_frequency,
        frequency_threshold=frequency_threshold,
        frequency_window_size=frequency_window_size
    )


if __name__ == "__main__":
    main()