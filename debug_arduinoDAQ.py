import h5py
import plotly.graph_objects as go
import numpy as np
import argparse

def plot_hdf5_laser_channel_as_square_wave(h5_file_path):
    """
    Plot only the 'LASER' channel from the HDF5 file as a square wave interactively using Plotly.

    Args:
        h5_file_path (str): Path to the HDF5 file.
    """
    try:
        # Open the HDF5 file
        with h5py.File(h5_file_path, 'r') as h5f:
            # Extract timestamps and check for the 'LASER' channel
            timestamps = h5f['timestamps'][:]
            channel_group = h5f['channel_data']

            if 'LASER' not in channel_group:
                print("The 'LASER' channel was not found in the HDF5 file.")
                return

            laser_data = channel_group['LASER'][:]
            
            # Create square wave style data
            time_steps = np.repeat(timestamps, 2)[1:]  # Duplicate timestamps for steps, drop the first
            values_steps = np.repeat(laser_data, 2)  # Duplicate values for steps

            # Prepare the plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=time_steps,
                y=values_steps,
                mode='lines',
                name='LASER'
            ))

            # Update the layout for better interactivity
            fig.update_layout(
                title="Interactive Plot of 'LASER' Channel as Square Wave",
                xaxis_title="Time (s)",
                yaxis_title="Laser Data",
                hovermode="x unified",
                template="plotly_dark",
            )

            # Display the plot
            fig.show()

    except Exception as e:
        print(f"An error occurred while plotting the 'LASER' channel: {e}")


def main():
    h5_file_path =  r"D:\test_output\241203_131302\241203_131302_test1\241203_131302_test1-ArduinoDAQ.h5"

    # Call the plotting function
    plot_hdf5_laser_channel_as_square_wave(h5_file_path)

if __name__ == "__main__":
    main()
