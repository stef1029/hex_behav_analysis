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

def DAQ_to_nwb(DAQ_h5_path: Path,
               scales_data: dict,
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
    Convert trilab data to NWB, reading DAQ data from an HDF5 file.
    :param DAQ_h5_path: Path to the HDF5 file containing DAQ data.
    :param scales_data: Dictionary containing scales data.
    :param session_ID: Session identifier (str).
    :param mouse_id: Mouse ID (str).
    :param video_directory: Path to the video directory.
    :param video_timestamps: Dictionary containing video timestamps.
    :param session_directory: Path to the session directory.
    :param session_metadata: Dictionary containing session metadata.
    :param session_description: Session description (str).
    :param experimenter: Name of the experimenter (str).
    :param institution: Name of the institution (str).
    :param lab: Name of the lab (str).
    :return: NWBFile object.
    """
    import h5py

    # Open the HDF5 file and read the data
    with h5py.File(DAQ_h5_path, 'r') as h5f:
        # Read the timestamps
        timestamps = np.array(h5f['timestamps'], dtype=np.float64)

        # Read the channel data
        channel_data = {}
        for channel_name in h5f['channel_data']:
            channel_data[channel_name] = np.array(h5f['channel_data'][channel_name], dtype=np.int8)

    # Convert to datetime object
    trialdate = datetime.strptime(session_ID[:13], '%y%m%d_%H%M%S')
    trialdate = trialdate.replace(tzinfo=tz.gettz('Europe/London'))

    # Set up NWB file
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=trialdate,
        experimenter=experimenter,
        institution=institution,
        lab=lab,
        experiment_description=f"phase:{session_metadata['behaviour_phase']}; "
                               f"rig:{session_metadata['rig_id']}; "
                               f"wait:{session_metadata['wait_duration']}; "
                               f"cue:{session_metadata['cue_duration']}"
    )

    # Set up subject info
    nwbfile.subject = Subject(
        subject_id=mouse_id,
        species='Mouse',
        weight=float(session_metadata['mouse_weight'])
    )

    # Loop through BUZZER and LED stimulus data types
    for i in ['BUZZER', 'LED_']:
        for j in range(1, 7):
            channel_name = i + str(j)
            if channel_name in channel_data:
                # Load data
                array = channel_data[channel_name]

                # Load epochs and epoch timestamps
                intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)

                # Create interval series
                interval_series = IntervalSeries(
                    name=channel_name,
                    timestamps=interval_timestamps,
                    data=intervals,
                    description='Intervals for ' + channel_name
                )

                # Add to NWB file
                nwbfile.add_stimulus(interval_series)

    for i in ['GO_CUE', 'NOGO_CUE']:
        if i in channel_data:
            # Load data
            array = channel_data[i]

            # Load epochs and epoch timestamps
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)

            # Create interval series
            interval_series = IntervalSeries(
                name=i,
                timestamps=interval_timestamps,
                data=intervals,
                description='Intervals for ' + i
            )

            # Add to NWB file
            nwbfile.add_stimulus(interval_series)

    # Add spotlight as timeseries to stimuli
    for j in range(1, 7):
        channel_name = 'SPOT' + str(j)
        if channel_name in channel_data:
            # Load data
            array = channel_data[channel_name]

            timeseries = TimeSeries(
                name=channel_name,
                data=array,
                timestamps=timestamps,
                unit='n.a',
                comments='Spotlight brightness represented between 1 and 0, '
                         'a low-passed version of the PWM wave that controlled the spotlight brightness.',
                description='Spotlight data at ' + channel_name
            )

            # Add to NWB file
            nwbfile.add_stimulus(timeseries)

    # Valve reward data
    for j in range(1, 7):
        channel_name = 'VALVE' + str(j)
        if channel_name in channel_data:
            # Load data
            array = channel_data[channel_name]

            # Load epochs and epoch timestamps
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)
            digital_events, digital_event_timestamps = intervals_to_digital_events(intervals, interval_timestamps)

            # Create timeseries
            timeseries = TimeSeries(
                name=channel_name,
                data=digital_events,
                timestamps=digital_event_timestamps,
                unit='s',
                comments='Reward amounts are measured by how long the valve is open for',
                description='Reward amount at ' + channel_name
            )

            # Add to NWB file
            nwbfile.add_stimulus(timeseries)

    # Loop through behavior data types
    # Sensor data
    i = 'SENSOR'
    for j in range(1, 7):
        channel_name = i + str(j)
        if channel_name in channel_data:
            # Load data
            array = channel_data[channel_name]

            # Load epochs and epoch timestamps
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array, HIGH=0)

            # Create interval series
            interval_series = IntervalSeries(
                name=channel_name,
                timestamps=interval_timestamps,
                data=intervals,
                description='Epochs for ' + channel_name
            )

            # Add to NWB file
            nwbfile.add_acquisition(interval_series)

    # Scales data
    # Load weights and timestamps
    weights = np.array(scales_data['weights'], dtype=np.float64)
    scales_timestamps = np.array(scales_data['timestamps'], dtype=np.float64)

    # Create timeseries
    timeseries = TimeSeries(
        name='scales',
        data=weights,
        timestamps=scales_timestamps,
        unit='g',
        comments=f"Threshold set to {scales_data['mouse_weight_threshold']}g. "
                 "Scales data not timestamp accurate; these are estimated times and are not synced with other devices.",
        description='Scales data'
    )

    # Add to NWB file
    nwbfile.add_acquisition(timeseries)

    # Load video data
    if not video_directory.exists():
        raise FileNotFoundError(f'Video file {video_directory} not found')

    # Make numpy array of timestamps
    numeric_keys = [key for key in video_timestamps.keys() if str(key).isdigit()]
    sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
    video_ts = np.array([video_timestamps[key] for key in sorted_keys], dtype=np.float64)

    # Create ImageSeries
    behaviour_video = ImageSeries(
        name='behaviour_video',
        external_file=['./' + video_directory.name],
        starting_frame=[0],
        format='external',
        timestamps=video_ts,
        unit='n.a',
        description='Behaviour top-down video'
    )

    # Add to NWB file
    nwbfile.add_acquisition(behaviour_video)

    # Save NWB file
    savepath = session_directory / (session_directory.stem + '.nwb')
    with NWBHDF5IO(savepath, 'w') as io:
        io.write(nwbfile)
