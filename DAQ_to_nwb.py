from pynwb import NWBHDF5IO, NWBFile, TimeSeries, load_namespaces, register_class
from pynwb.misc import IntervalSeries
from pynwb.file import Subject, LabMetaData
from pynwb.image import ImageSeries
from hdmf.utils import docval, get_docval, popargs

from datetime import datetime
from dateutil import tz
from uuid import uuid4
import numpy as np
from pathlib import Path
import json
import shutil

# function takes timestamps and on/off and returns intervals for NWB
def timeseries_to_intervals(timestamps, signal, HIGH = 1, filter = False):
    """
    Convert timestamps and on/off signal to NWB epochs
    :param timestamps: timestamps of signal (list)
    :param signal: on/off signal (list)
    :return: NWB intervals: 1 for on times, -1 for off times (np.array)
    :return: timestamps (np.array)
    """
    timestamps = np.array(timestamps)
    signal = np.array(signal)

    if HIGH == 0:
        HIGH = -1

    if filter:      # if filter true, filter for times longer than 50ms:

        duration_threshold = 50
        
        # Compute difference array to find changes in state
        diff = HIGH * np.diff(signal, prepend=0)

        # Find indices of on and off times
        change_indices = np.where(diff != 0)[0]
        
        # Ensure even number of changes (start and end for each event)
        if len(change_indices) % 2 != 0:
            change_indices = np.append(change_indices, len(signal) - 1)

        # Calculate event durations
        start_times = timestamps[change_indices[::2]]
        end_times = timestamps[change_indices[1::2]]
        durations = end_times - start_times

        # Filter events based on duration
        long_event_indices = np.where(durations >= duration_threshold / 1000.0)[0]  # Convert ms to seconds if needed

        # Get timestamps and intervals for long events
        long_events_start_indices = change_indices[::2][long_event_indices]
        long_events_end_indices = change_indices[1::2][long_event_indices]
        
        interval_timestamps = np.concatenate((timestamps[long_events_start_indices], timestamps[long_events_end_indices]))
        intervals = np.concatenate((HIGH * np.ones_like(long_events_start_indices), -HIGH * np.ones_like(long_events_end_indices)))

        # Sort the timestamps and intervals to maintain chronological order
        sort_indices = np.argsort(interval_timestamps)
        interval_timestamps = interval_timestamps[sort_indices]
        intervals = intervals[sort_indices]

        return intervals, interval_timestamps

    else:       # if filter false, do not filter for times:
        
        # compute difference array
        diff = HIGH * np.diff(signal, prepend=0)

        # find indices of on and off times
        indices = np.where(diff != 0)[0]

        # load timestamps and on/off signal
        interval_timestamps = timestamps[indices]
        intervals = diff[indices]

        return intervals, interval_timestamps

# fuunction converts intervals to digital events with length of time
def intervals_to_digital_events(intervals, interval_timestamps):
    """
    Convert intervals to digital events
    :param intervals: intervals (np.array)
    :param interval_timestamps: timestamps of intervals (np.array)
    :return: digital events (np.array)
    :return: timestamps (np.array)
    """

    # make sure it starts with an on time and ends with and off time
    if intervals.shape[0] > 0:
        if intervals[0] == -1:
            intervals = intervals[1:]
            interval_timestamps = interval_timestamps[1:]
        if intervals[-1] == 1:
            intervals = intervals[:-1]
            interval_timestamps = interval_timestamps[:-1]

    # find lengths of on times
    digital_events = interval_timestamps[intervals == -1] - interval_timestamps[intervals == 1]
    timestamps = interval_timestamps[intervals == 1]
    
    return digital_events, timestamps

# function to convert trilab data to NWB
def DAQ_to_nwb(DAQ_dict: dict, 
               session_ID: str, 
               mouse_id: str, 
               video_directory: Path, 
               video_timestamps: dict,
               session_directory: Path,
               session_metadata,
               session_description: str, 
               experimenter: str,
               institution: str, 
               lab: str) -> NWBFile:
    """
    Convert trilab data to NWB
    :In: DAQ_dict: dictionary containing DAQ data from preliminary processing
    :param session_ID: datetime object of trial datetime
    :param subject_id: subject ID (str)
    """

    # convert to datetime object
    trialdate = datetime.strptime(session_ID[:13], '%y%m%d_%H%M%S')
    trialdate = trialdate.replace(tzinfo=tz.gettz('Europe/London'))

    data = DAQ_dict

    # set up NWB file
    nwbfile = NWBFile(
        session_description = session_description,
        identifier = str(uuid4()),
        session_start_time = trialdate,
        experimenter = experimenter,
        institution = institution,
        lab = lab,
        experiment_description=f"phase:{session_metadata['behaviour_phase']}; rig:{session_metadata['rig_id']}; wait:{session_metadata['wait_duration']}; cue:{session_metadata['cue_duration']}"
    )

    # set up subject info
    nwbfile.subject = Subject(
        subject_id = mouse_id,
        species = 'Mouse',
        weight=float(session_metadata['mouse_weight'])
    )

    # load timestamps array
    timestamps = np.array(data['timestamps'], dtype=np.float64)

    # loop through Buzzer and LED stimulus data types

    for i in ['BUZZER','LED_']:

        for j in range(1,7):

            # load data
            array = np.array(data[i + str(j)], dtype=np.int8)
            
            # load epochs and epoch timestamps
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)

            # create interval series
            interval_series = IntervalSeries(
                name=i + str(j),
                timestamps=interval_timestamps,
                data=intervals,
                description='Intervals for ' + i + str(j)
            )

            # add to NWB file
            nwbfile.add_stimulus(interval_series)

    for i in ['GO_CUE','NOGO_CUE']:

        # load data
        array = np.array(data[i], dtype=np.int8)
        
        # load epochs and epoch timestamps
        intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)

        # create interval series
        interval_series = IntervalSeries(
            name=i,
            timestamps=interval_timestamps,
            data=intervals,
            description='Intervals for ' + i 
        )

        # add to NWB file
        nwbfile.add_stimulus(interval_series)

    # add spotlight as timeseries to stimuli:
    for j in range(1,7):

        # load data:
        array = np.array(data['SPOT' + str(j)], dtype=np.int8)

        timeseries = TimeSeries(
            name='SPOT' + str(j),
            data=array,
            timestamps=timestamps,
            unit='n.a',
            comments='Spotlight brightness represented between 1 and 0, a low passed version of the pwm wave that controlled the spotlight brightness.',
            description='Spotlight data at SPOT' + str(j)
        )

        # add to nwbfile:
        nwbfile.add_stimulus(timeseries)

    # valve reqard data
    for j in range(1,7):

        # load data
        array = np.array(data['VALVE' + str(j)], dtype=np.int8)

        # load epochs and epoch timestamps
        intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)
        digital_events, digital_event_timestamps = intervals_to_digital_events(intervals, interval_timestamps)

        # create timeseries
        timeseries = TimeSeries(
            name='VALVE' + str(j),
            data=digital_events,
            timestamps=digital_event_timestamps,
            unit='s',
            comments='Reward amounts are measured by how long the valve is open for',
            description='Reward amount at VALVE' + str(j)
        )

        # add to NWB file
        nwbfile.add_stimulus(timeseries)



    ## loop through behaviour data types

    # sensor data
    i = 'SENSOR'
    for j in range(1,7):

        # load data
        array = np.array(data[i + str(j)], dtype=np.int8)
        
        # load epochs and epoch timestamps
        intervals, interval_timestamps = timeseries_to_intervals(timestamps, array, HIGH = 0)

        # create interval series
        interval_series = IntervalSeries(
            name=i + str(j),
            timestamps=interval_timestamps,
            data=intervals,
            description='Epochs for ' + i + str(j)
        )

        # add to NWB file
        nwbfile.add_acquisition(interval_series)

    # scales data
    # load weights
    weights = np.array(data['scales_data']['weights'], dtype=np.float64)

    # load timestamps
    timestamps = np.array(data['scales_data']['timestamps'], dtype=np.float64)

    # create timeseries
    timeseries = TimeSeries(
        name='scales',
        data=weights,
        timestamps=timestamps,
        unit='g',
        comments=f'''Threshold set to {data['scales_data']['mouse_weight_threshold']}g.
                    Scales data not timestamp accurate, these are estimated times,
                    and so are not synced with other devices.''',
        description='Scales data'
    )

    # add to NWB file
    nwbfile.add_acquisition(timeseries)

    # load video data
    if not video_directory.exists():
        raise FileNotFoundError(f'Video file {video_directory} not found')

    # make numpy array of timestamps
    numeric_keys = [key for key in video_timestamps.keys() if str(key).isdigit()]
    sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
    timestamps = np.array([video_timestamps[key] for key in sorted_keys], dtype=np.float64)

    frame_IDs_saved = video_timestamps["frame_IDs_saved?"]
    pulse_times_simulated = video_timestamps["pulse_times_simulated?"]
    simulated_frames = video_timestamps["simulated_frames"]
    no_dropped_frames = video_timestamps["no_dropped_frames"]

    comments_text = f"frame_IDs_saved: {frame_IDs_saved}, " \
                    f"pulse_times_simulated: {pulse_times_simulated}, " \
                    f"simulated_frames: {simulated_frames}, " \
                    f"no_dropped_frames: {no_dropped_frames}"

    # create ImageSeries
    behaviour_video = ImageSeries(
        name='behaviour_video',
        external_file=['./' + video_directory.name],
        starting_frame=[0],
        format='external',
        timestamps=timestamps,
        unit='n.a',
        description='Behaviour top down video',
        comments=comments_text
    )

    # add to NWB file
    nwbfile.add_acquisition(behaviour_video)

    # save NWB file
    savepath = session_directory / (session_directory.stem + '.nwb')
    with NWBHDF5IO(savepath, 'w') as io:
        io.write(nwbfile)

