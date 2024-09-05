#%%

import pynwb
from pathlib import Path
import nwbwidgets as nwbw

# Path to the NWB file
nwb_path = Path("/cephfs2/srogers/March_training/240325_161028/240325_161055_wtjx287-5d/240325_161055_wtjx287-5d.nwb")
#%%
# Open the NWB file
# with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
#     nwbfile = io.read()

#     # Accessing SENSOR1 data
#     # The exact location and name of SENSOR1 data within the NWB file will depend on how it was stored
#     # Checking in acquisition
#     if 'SENSOR1' in nwbfile.acquisition:
#         sensor1_data = nwbfile.acquisition['SENSOR1']
#     # If not in acquisition, you may need to check in stimulus or other groups, depending on the file's structure
#     elif 'SENSOR1' in nwbfile.stimulus:
#         sensor1_data = nwbfile.stimulus['SENSOR1']
#     else:
#         print("SENSOR1 data not found in expected groups.")
#         sensor1_data = None

#     # Print an example of the data
#     if sensor1_data:
#         print("SENSOR1 Data:")
#         print("Data:", sensor1_data.data[:10])  # Print first 10 data points as an example
#         print("Timestamps:", sensor1_data.timestamps[:10])  # Print first 10 timestamps

io = pynwb.NWBHDF5IO(str(nwb_path), 'r')
nwbfile = io.read()

nwbw.nwb2widget(nwbfile)
# %%
