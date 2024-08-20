#%%


import json
from pathlib import Path
import os
import matplotlib.pyplot as plt

def find_file(directory, tag):
    file_found = False
    for file in directory.glob('*'):
        if not file_found:
            if tag in file.name:
                file_found = True
                return file
    if not file_found:
        raise Exception(f"'{tag}' not found in {directory}")
    

onedrive = Path(os.getenv('OneDrive'))
# behaviour_files =Path(os.getenv('BehaviourFiles'))

# test_files = onedrive / r"01 - PhD at LMB\Coding projects\231101 - New analysis pipeline\231205_213113_test"

test_files = Path(r"V:\New analysis pipeline\231207_164535_wtjx285-2d")

processed_DAQ_data = find_file(test_files, 'processed')
print(str(processed_DAQ_data))
print("Loading data...")

with open(processed_DAQ_data, 'r') as f:
    DAQ_data = json.load(f)
print("Data loaded")

#%%

channels = DAQ_data.keys()
# print(channels)
"""
find led events. this will tell men when trials are starting.
then find what happens next, a valve or a spotlight?
then, within those timestamps, collect all the sensor activations.
if its a valve, one of them will be for the led port, if not then find the port which set off the spotlight.
"""


def get_sensor_events():
    sensor_events = [[[], []] for i in range(6)]

    for i, timestamp in enumerate(DAQ_data["timestamps"]):
        if i != 0:
            for j in range(6):
                if DAQ_data[f"SENSOR{j+1}"][i] == '0' and DAQ_data[f"SENSOR{j+1}"][i-1] == '1':
                    sensor_events[j][0].append(timestamp)
                if DAQ_data[f"SENSOR{j+1}"][i] == '1' and DAQ_data[f"SENSOR{j+1}"][i-1] == '0':
                    sensor_events[j][1].append(timestamp)


    # check for events which overlap with start or end of recording and remove:
    # list lengths checks for non matching lengths of onset and offset events
    list_lengths = [[len(sensor_events[i][0]), len(sensor_events[i][1])] for i in range(6)]

    for i in range(6):
        if list_lengths[i][0] != list_lengths[i][1]:
            # check the first and last values of the sensor events to see if they are 0 or 1.
            first_reading = DAQ_data[f"SENSOR{i+1}"][0]
            last_reading = DAQ_data[f"SENSOR{i+1}"][-1]
            # if they're zero, then it means the sensor was being activated at the start or end of the recording.
            if first_reading == '0':
                sensor_events[i][1].pop(0)
            if last_reading == '0':
                sensor_events[i][0].pop(-1)
    
    # zip the lists together into a list of tuples:
    sensor_events = [list(zip(sensor_events[i][0], sensor_events[i][1])) for i in range(6)]

    return sensor_events




print(get_sensor_events())
# %%


def find_trials():
    # sensor_events are sometimes immediately followed by a valve event.
    pass