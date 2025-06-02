from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.misc import IntervalSeries
from pynwb.file import Subject
from pynwb.image import ImageSeries
from hdmf.utils import docval, get_docval, popargs

from datetime import datetime
from dateutil import tz
from uuid import uuid4
import numpy as np
from pathlib import Path
import json
import shutil
import h5py

# -------------------------------------------------------------------------
# (1) Helpers to detect pulses and intervals
# -------------------------------------------------------------------------
def detect_rising_edges(signal, timestamps, threshold=0.5):
    """
    Identify the indices of rising edges (TTL pulses) in the given signal.
    A rising edge is where signal[i] < threshold and signal[i+1] >= threshold.
    Returns the indices of the rising edges.
    """
    signal = np.asarray(signal)
    if len(signal) == 0:
        return np.array([])

    above_thresh = signal >= threshold
    rising = (above_thresh[1:] == True) & (above_thresh[:-1] == False)
    edges = np.where(rising)[0] + 1  # +1 offset because we shifted by 1
    return edges

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
        
        interval_timestamps = np.concatenate((timestamps[long_events_start_indices],
                                              timestamps[long_events_end_indices]))
        intervals = np.concatenate((HIGH * np.ones_like(long_events_start_indices),
                                    -HIGH * np.ones_like(long_events_end_indices)))

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

def intervals_to_digital_events(intervals, interval_timestamps):
    """
    Convert intervals to digital events (duration-based).
    :param intervals: intervals (np.array)
    :param interval_timestamps: timestamps of intervals (np.array)
    :return: digital_events (np.array)
    :return: timestamps (np.array)
    """

    # make sure it starts with an on time and ends with an off time
    if intervals.shape[0] > 0:
        if intervals[0] == -1:
            intervals = intervals[1:]
            interval_timestamps = interval_timestamps[1:]
        if intervals.shape[0] > 0 and intervals[-1] == 1:
            intervals = intervals[:-1]
            interval_timestamps = interval_timestamps[:-1]

    # find lengths of on times
    on_times = interval_timestamps[intervals == 1]
    off_times = interval_timestamps[intervals == -1]

    # If they mismatch for some reason, safely handle
    n_on = len(on_times)
    n_off = len(off_times)
    if n_on > n_off:
        # Drop extra on_times
        on_times = on_times[:n_off]
    elif n_off > n_on:
        # Drop extra off_times
        off_times = off_times[:n_on]

    digital_events = off_times - on_times
    return digital_events, on_times

# -------------------------------------------------------------------------
# (2) Main DAQ_to_nwb function with new logic to handle missing pulses
# -------------------------------------------------------------------------
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
               lab: str,
               max_frame_id: int=None  # <-- new optional parameter
               ) -> NWBFile:
    """
    Convert Trilab data to NWB, reading DAQ data from an HDF5 file.
    Now checks whether the first camera pulse is too close to 0 (meaning pulses were truncated),
    and if so, shifts the video timestamps backward in time.

    :param max_frame_id: The highest frame number we expected. If the number of detected pulses
                         is smaller than max_frame_id, we assume some pulses were chopped off.
    """
    # ---------------------------------------------------------------------
    # (A) Load the Arduino DAQ data
    # ---------------------------------------------------------------------
    with h5py.File(DAQ_h5_path, 'r') as h5f:
        timestamps = np.array(h5f['timestamps'], dtype=np.float64)

        # Read the channel data
        channel_data = {}
        for channel_name in h5f['channel_data']:
            channel_data[channel_name] = np.array(h5f['channel_data'][channel_name], dtype=np.int8)

    # ---------------------------------------------------------------------
    # (B) If "CAMERA" channel is present, detect pulses & see if we should shift
    # ---------------------------------------------------------------------
    if max_frame_id is not None and "CAMERA" in channel_data:
        camera_signal = channel_data["CAMERA"]
        edges = detect_rising_edges(camera_signal, timestamps, threshold=0.5)

        if len(edges) > 0:
            # Time of the first pulse
            first_pulse_time = timestamps[edges[0]]
            # Time of the last pulse
            last_pulse_time = timestamps[edges[-1]]
            # Total recording duration
            total_duration = timestamps[-1]
            # Number of pulses
            detected_pulses = len(edges)

            # Heuristic check: if the first pulse is under [start minimum]ms from 0,
            # we suspect we've missed pulses that happened prior to DAQ start
            start_minimum = 0.1  # seconds
            if first_pulse_time < start_minimum:
                missing_frames = max_frame_id - detected_pulses
                print(max_frame_id, detected_pulses)
                if missing_frames > 0:
                    # Calculate framerate from the pulses we do have
                    # Here, we use the median difference between consecutive camera pulses
                    # as the "typical" frame interval. Alternatively, you can do total range / count.
                    pulse_times = timestamps[edges]
                    if len(pulse_times) > 1:
                        median_dt = np.median(np.diff(pulse_times))  # seconds per frame
                        frame_rate = 1.0 / median_dt
                    else:
                        # Fallback if we can't compute median
                        frame_rate = 30.0  # some default, e.g. 30 FPS

                    shift_time = missing_frames / frame_rate
                    print(f"  [CAMERA WARNING] The first camera pulse occurs at {first_pulse_time*1000:.2f} ms "
                        f"(<{start_minimum*1000:.2f} ms). Likely truncated pulses. "
                        f"Shifting video timestamps backward by {shift_time:.3f} s.\n")

                    # Shift all video timestamps backward by shift_time
                    # (only if it doesn't push them below 0 in a problematic way)
                    # You can decide how to handle a negative result; here we just shift anyway.
                    for key in video_timestamps.keys():
                        # If it's numeric
                        if str(key).isdigit():
                            video_timestamps[key] = float(video_timestamps[key]) - shift_time
            else:
                print(f"  [CAMERA INFO] First camera pulse occurs at {first_pulse_time*1000:.2f} ms.")
            
            # New check for the end of recording
            end_minimum = 0.1  # seconds
            time_to_end = total_duration - last_pulse_time
            if time_to_end < end_minimum:
                print(f"  [CAMERA WARNING] The last camera pulse occurs at {last_pulse_time*1000:.2f} ms, "
                    f"which is only {time_to_end*1000:.2f} ms from the end of recording "
                    f"(<{end_minimum*1000:.2f} ms). Likely truncated pulses at the end.")
                
                # Here you could add logic similar to the start check if you want to adjust timestamps
                # For example, estimate if there are additional frames we might have missed
                # This would depend on your specific requirements
            else:
                print(f"  [CAMERA INFO] Last camera pulse occurs at {last_pulse_time*1000:.2f} ms, "
                    f"{time_to_end*1000:.2f} ms from end of recording.")
    # ---------------------------------------------------------------------
    # (C) Prepare an NWBFile with relevant metadata
    # ---------------------------------------------------------------------
    trialdate = datetime.strptime(session_ID[:13], '%y%m%d_%H%M%S')
    trialdate = trialdate.replace(tzinfo=tz.gettz('Europe/London'))

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=trialdate,
        experimenter=experimenter,
        institution=institution,
        lab=lab,
        experiment_description=(
            f"phase:{session_metadata['behaviour_phase']}; "
            f"rig:{session_metadata['rig_id']}; "
            f"wait:{session_metadata['wait_duration']}; "
            f"cue:{session_metadata['cue_duration']}"
        )
    )

    nwbfile.subject = Subject(
        subject_id=mouse_id,
        species='Mouse',
        weight=float(session_metadata['mouse_weight'])
    )

    # ---------------------------------------------------------------------
    # (D) Add relevant channels as stimuli or acquisitions
    # ---------------------------------------------------------------------
    # Example for BUZZER & LED_...
    for i in ['BUZZER', 'LED_']:
        for j in range(1, 7):
            channel_name = i + str(j)
            if channel_name in channel_data:
                array = channel_data[channel_name]
                intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)
                interval_series = IntervalSeries(
                    name=channel_name,
                    timestamps=interval_timestamps,
                    data=intervals,
                    description='Intervals for ' + channel_name
                )
                nwbfile.add_stimulus(interval_series)

    # Example for GO_CUE & NOGO_CUE
    for i in ['GO_CUE', 'NOGO_CUE']:
        if i in channel_data:
            array = channel_data[i]
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)
            interval_series = IntervalSeries(
                name=i,
                timestamps=interval_timestamps,
                data=intervals,
                description='Intervals for ' + i
            )
            nwbfile.add_stimulus(interval_series)

    # Add spotlight channels
    for j in range(1, 7):
        channel_name = 'SPOT' + str(j)
        if channel_name in channel_data:
            array = channel_data[channel_name]
            timeseries = TimeSeries(
                name=channel_name,
                data=array,
                timestamps=timestamps,
                unit='n.a',
                comments=(
                    'Spotlight brightness represented between 1 and 0, '
                    'a low-passed version of the PWM wave that controlled the spotlight.'
                ),
                description='Spotlight data at ' + channel_name
            )
            nwbfile.add_stimulus(timeseries)

    # Valve reward data
    for j in range(1, 7):
        channel_name = 'VALVE' + str(j)
        if channel_name in channel_data:
            array = channel_data[channel_name]
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array)
            digital_events, digital_event_timestamps = intervals_to_digital_events(intervals, interval_timestamps)
            timeseries = TimeSeries(
                name=channel_name,
                data=digital_events,
                timestamps=digital_event_timestamps,
                unit='s',
                comments='Reward amounts measure how long the valve is open',
                description='Reward at ' + channel_name
            )
            nwbfile.add_stimulus(timeseries)

    # Sensor data
    i = 'SENSOR'
    for j in range(1, 7):
        channel_name = i + str(j)
        if channel_name in channel_data:
            array = channel_data[channel_name]
            intervals, interval_timestamps = timeseries_to_intervals(timestamps, array, HIGH=0)
            interval_series = IntervalSeries(
                name=channel_name,
                timestamps=interval_timestamps,
                data=intervals,
                description='Epochs for ' + channel_name
            )
            nwbfile.add_acquisition(interval_series)

    # ---------------------------------------------------------------------
    # (E) Scales data
    # ---------------------------------------------------------------------
    weights = np.array(scales_data['weights'], dtype=np.float64)
    scales_timestamps = np.array(scales_data['timestamps'], dtype=np.float64)
    scales_ts = TimeSeries(
        name='scales',
        data=weights,
        timestamps=scales_timestamps,
        unit='g',
        comments=(
            f"Threshold set to {scales_data['mouse_weight_threshold']}g. "
            "Scales data not timestamp-accurate; these are approximate times not synced with other devices."
        ),
        description='Scales data'
    )
    nwbfile.add_acquisition(scales_ts)

    # ---------------------------------------------------------------------
    # (F) Video data (potentially already shifted if pulses were missing)
    # ---------------------------------------------------------------------
    print(f"  [VIDEO INFO] Adding video data from {video_directory}")
    if not video_directory.exists():
        raise FileNotFoundError(f'Video file {video_directory} not found')

    # Sort keys and make a numpy array of timestamps
    numeric_keys = [key for key in video_timestamps.keys() if str(key).isdigit()]
    sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
    video_ts = np.array([video_timestamps[key] for key in sorted_keys], dtype=np.float64)

    # If max_frame_id is provided, ensure we don't exceed it
    if max_frame_id is not None:
        if len(video_ts) > max_frame_id + 1:  # +1 because frame IDs start at 0
            print(f"  [VIDEO INFO] Truncating video timestamps from {len(video_ts)} to {max_frame_id + 1} frames")
            video_ts = video_ts[:max_frame_id + 1]
            
    # Additional safety check: ensure we have valid timestamps
    if len(video_ts) == 0:
        raise ValueError("No valid video timestamps found")
        
    print(f"  [VIDEO INFO] Final video timestamp array length: {len(video_ts)}")
    print(f"  [VIDEO INFO] Video timestamp range: {video_ts[0]:.3f}s to {video_ts[-1]:.3f}s")

    behaviour_video = ImageSeries(
        name='behaviour_video',
        external_file=['./' + video_directory.name],
        starting_frame=[0],
        format='external',
        timestamps=video_ts,  # This should now be the right length
        unit='n.a',
        description='Behaviour top-down video'
    )
    nwbfile.add_acquisition(behaviour_video)

    # ---------------------------------------------------------------------
    # (G) Save NWB file
    # ---------------------------------------------------------------------
    savepath = session_directory / (session_directory.stem + '.nwb')
    with NWBHDF5IO(savepath, 'w') as io:
        io.write(nwbfile)

    return nwbfile
