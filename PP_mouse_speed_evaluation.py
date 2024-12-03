import matplotlib.pyplot as plt
import numpy as np
import math
from pynwb import NWBHDF5IO

def evaluate_speed(sessions, 
                        title='title', 
                        start_angle='all'):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot

    This function takes the list of sessions to use and extracts the trials per mouse.
    Split trials into successful and unsuccessful trials.
    Get mouse heading at cue offset time, and calculate angle to correct port.
    Then bin the trials into groups based on this angle. 
    For successful trials, this might cause a large number of trials in the 0 degree range.
    For unsuccessful trials, this might show a reduced number in the 0 degree range.
    """

    mice = {}
    # for each session that I give it:
    for session in sessions:
        # grab the mouse ID
        mouse_id = session.session_dict['mouse_id']
        # grab the trials:
        trials = session.trials

        if mouse_id not in mice:
            mice[mouse_id] = {'trials': [], 'successful_trials': [], 'unsuccessful_trials': []}
        # store the trials in the mouse dictionary:
        mice[mouse_id]['trials'] += trials
        mice[mouse_id]['rig_id'] = session.rig_id
        mice[mouse_id]['session'] = session

        # now start creating the binned trials info:
        num_bins = 6
        bin_size = round(180 / num_bins)

        # for each mouse, creates a binned trials dictionary, which contains all the trials where the mouse had the cue angle within the bin range.
        bins = {i: {'all_trials': [], 'successful_trials': [], 'unsuccessful_trials': [], 'timeout': []} for i in range(0, 180, bin_size)}

        # for each trial, find which bin it goes in:
        for trial in trials:
            if trial['turn_data'] is not None:
                angle = abs(trial['turn_data']['cue_presentation_angle'])
                for bin in bins:
                    if angle < bin + bin_size and angle >= bin:
                        bins[bin]['all_trials'].append(trial)
        mice[mouse_id]['binned_trials'] = bins

        # then, using the binned trials, also find the successful and unsuccessful ones:
        # for each angle bin:
        for key, bin in mice[mouse_id]['binned_trials'].items():
            # for each trial:
            for trial in bin['all_trials']:
                if trial['cue_end'] is not None and trial['DLC_data'] is not None:
                    if trial['next_sensor'] != {}:
                        if trial['correct_port'] == trial['next_sensor']['sensor_touched']:
                            bin['successful_trials'].append(trial)
                        else:
                            bin['unsuccessful_trials'].append(trial)
                    # else:
                    #     bin['unsuccessful_trials'].append(trial)
                    #     bin['timeout'].append(trial)
        
        # now ech mouse has its trials sorted into bins based on angle, and success.

    # for each trial in successful trials, get the correct port, get the dlc data,
    # get cue offset time, and then grab the head angle at that time.
    # then use the calibrated angles to get the angle that mouse was facing relative to the correct port.
    # Append this to the correct bin.
    global sample_times
    global speed_times
    global acceleration_times
    sample_times = []
    speed_times = []
    acceleration_times = []

    def get_data(trials, mouse_id):

        min_trial_duration = 0.3
        time_before_cue = 2
        smoothing = 1

        speed_arrays = []
        acceleration_arrays = []
        session = mice[mouse_id]['session']
        full_dlc_data = session.DLC_coords
        # with NWBHDF5IO(session.nwb_file_path, 'r') as io:
        #     nwbfile = io.read()
        #     video_timestamps = nwbfile.acquisition['behaviour_video'].timestamps[:]
        video_timestamps = session.video_timestamps
        # print(mice[mouse_id]['rig_id'])
        for trial in trials:
            dlc_data = trial['DLC_data']
            start_time = trial['cue_start']
            framerate = 30

            # find start of trial as index in dlc data
            start_index = np.searchsorted(video_timestamps, start_time)

            if trial['next_sensor'].get('sensor_start') is not None:
                # print('hello')
                end_time = trial['next_sensor']['sensor_start']

                trial_duration = end_time - start_time

                if trial_duration > min_trial_duration:

                    num_frames_after = int(min_trial_duration * framerate)
                    num_frames_before = int(time_before_cue * framerate)
                    dlc_data = full_dlc_data.loc[start_index - num_frames_before:start_index + num_frames_after - 1]

                    global sample_times
                    global speed_times
                    global acceleration_times
                    sample_times = []
                    for i in np.arange(-time_before_cue, min_trial_duration, 1/framerate):
                        sample_times.append(i)
                    speed_times = [time + 1/(2*framerate) for time in sample_times[:-1]]
                    acceleration_times = sample_times[1:-1]

                    # Extract midpoint coordinates
                    left_ear_x, left_ear_y = dlc_data[('left_ear', 'x')].to_numpy(), dlc_data[('left_ear', 'y')].to_numpy()
                    
                    # Calculate speeds using Euclidean distance between consecutive frames
                    distances = np.sqrt(np.diff(left_ear_x)**2 + np.diff(left_ear_y)**2)
                    speeds = distances * framerate  # Convert to units per second
                    
                    # Calculate accelerations as the change in speed between consecutive frames
                    accelerations = np.diff(speeds) * framerate  # Convert to units per second squared
                    if len(speeds) == len(speed_times):
                        speed_arrays.append(speeds)
                        acceleration_arrays.append(accelerations)

        if len(speed_arrays) > 0:
            speeds = np.mean(np.vstack(speed_arrays), axis=0)
            accelerations = np.mean(np.vstack(acceleration_arrays), axis=0)
        
            return speeds, accelerations
    
        else:
            return None, None


    success_data = {'data': {'speeds': {}, 'accelerations': {}},
                    'mean': {'speeds': {}, 'accelerations': {}},
                    'sem': {'speeds': {}, 'accelerations': {}}, 
                    'sd': {'speeds': {}, 'accelerations': {}}}
    unsuccessful_data = {'data': {'speeds': {}, 'accelerations': {}},
                         'mean': {'speeds': {}, 'accelerations': {}}, 
                         'sem': {'speeds': {}, 'accelerations': {}}, 
                         'sd': {'speeds': {}, 'accelerations': {}}}
    
    n_numbers = {'success': {}, 'fail': {}, 'timeout': {}}

    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heding angles at cue offset time.
        for key, bin in mice[mouse]['binned_trials'].items():
            successful_trials = bin['successful_trials']
            unsuccessful_trials = bin['unsuccessful_trials']
            timeout_trials = bin['timeout']

            speeds, accelerations = get_data(successful_trials, mouse)
            if speeds is not None and accelerations is not None:
                if key not in success_data['data']['speeds']:
                    success_data['data']['speeds'][key] = speeds
                if key in success_data['data']['speeds']:
                    success_data['data']['speeds'][key] = np.vstack((success_data['data']['speeds'][key], speeds))
                if key not in success_data['data']['accelerations']:
                    success_data['data']['accelerations'][key] = accelerations
                if key in success_data['data']['accelerations']:
                    success_data['data']['accelerations'][key] = np.vstack((success_data['data']['accelerations'][key], accelerations))


            speeds, accelerations = get_data(unsuccessful_trials, mouse)
            if speeds is not None and accelerations is not None:
                if key not in unsuccessful_data['data']['speeds']:
                    unsuccessful_data['data']['speeds'][key] = speeds
                if key in unsuccessful_data['data']['speeds']:
                    unsuccessful_data['data']['speeds'][key] = np.vstack((unsuccessful_data['data']['speeds'][key], speeds))
                if key not in unsuccessful_data['data']['accelerations']:
                    unsuccessful_data['data']['accelerations'][key] = accelerations
                if key in unsuccessful_data['data']['accelerations']:
                    unsuccessful_data['data']['accelerations'][key] = np.vstack((unsuccessful_data['data']['accelerations'][key], accelerations))


            n_success = len(successful_trials)
            n_fail = len(unsuccessful_trials)
            n_timeout = len(timeout_trials)

            if key not in n_numbers['success']:
                n_numbers['success'][key] = [n_success]
            else:
                n_numbers['success'][key].append(n_success)
            if key not in n_numbers['fail']:
                n_numbers['fail'][key] = [n_fail]
            else:
                n_numbers['fail'][key].append(n_fail)
            if key not in n_numbers['timeout']:
                n_numbers['timeout'][key] = [n_timeout]
            else:
                n_numbers['timeout'][key].append(n_timeout)

                

    for key, array_stack in success_data['data']['speeds'].items():
        # Sum along rows to combine all arrays into one
        success_data['mean']['speeds'][key] = np.mean(array_stack, axis=0)  
        success_data['sem']['speeds'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        success_data['sd']['speeds'][key] = np.std(array_stack, axis=0)

    for key, array_stack in success_data['data']['accelerations'].items():
        # Sum along rows to combine all arrays into one
        success_data['mean']['accelerations'][key] = np.mean(array_stack, axis=0)  
        success_data['sem']['accelerations'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        success_data['sd']['speeds'][key] = np.std(array_stack, axis=0)


    for key, array_stack in unsuccessful_data['data']['speeds'].items():
        # Sum along rows to combine all arrays into one
        unsuccessful_data['mean']['speeds'][key] = np.mean(array_stack, axis=0)  
        unsuccessful_data['sem']['speeds'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        unsuccessful_data['sd']['speeds'][key] = np.std(array_stack, axis=0)

    for key, array_stack in unsuccessful_data['data']['accelerations'].items():
        # Sum along rows to combine all arrays into one
        unsuccessful_data['mean']['accelerations'][key] = np.mean(array_stack, axis=0)  
        unsuccessful_data['sem']['accelerations'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        unsuccessful_data['sd']['accelerations'][key] = np.std(array_stack, axis=0)

    for key, list in n_numbers['success'].items():
        n_numbers['success'][key] = {'mean': np.mean(list),
                                     'sem': np.std(list) / np.sqrt(len(list)),
                                     'sd': np.std(list)}
        n_numbers['fail'][key] = {'mean': np.mean(n_numbers['fail'][key]),
                                    'sem': np.std(n_numbers['fail'][key]) / np.sqrt(len(n_numbers['fail'][key])),
                                    'sd': np.std(n_numbers['fail'][key])}
        n_numbers['timeout'][key] = {'mean': np.mean(n_numbers['timeout'][key]),
                                    'sem': np.std(n_numbers['timeout'][key]) / np.sqrt(len(n_numbers['timeout'][key])),
                                    'sd': np.std(n_numbers['timeout'][key])}

    # peak_accelerations = {}
    # # for each acceleration array, find the peak acceleration after time 0 and store it:

    # # Assume acceleration_times array and corresponding acceleration data arrays are correctly synchronized
    # zero_time_index = next(i for i, t in enumerate(acceleration_times) if t >= 0)  # Find index where time is zero or the first positive time

    # # Calculate and store peak accelerations and their times for each bin and each outcome category (successful, unsuccessful)
    # for key, array_stack in success_data['data']['accelerations'].items():
    #     if array_stack.shape[1] > zero_time_index:  # Ensure there is data after time zero
    #         peak_indices = np.argmax(array_stack[:, zero_time_index:], axis=1) + zero_time_index
    #         peak_accelerations_success = array_stack[np.arange(array_stack.shape[0]), peak_indices]
    #         peak_times_success = np.array(acceleration_times)[peak_indices]

    #         if key not in peak_accelerations:
    #             peak_accelerations[key] = {}

    #         peak_accelerations[key]['success'] = {
    #             'values': np.mean(peak_accelerations_success),
    #             'times': np.mean(peak_times_success)
    #         }

    # for key, array_stack in unsuccessful_data['data']['accelerations'].items():
    #     if array_stack.shape[1] > zero_time_index:
    #         peak_indices = np.argmax(array_stack[:, zero_time_index:], axis=1) + zero_time_index
    #         peak_accelerations_unsuccessful = array_stack[np.arange(array_stack.shape[0]), peak_indices]
    #         peak_times_unsuccessful = np.array(acceleration_times)[peak_indices]

    #         if key not in peak_accelerations:
    #             peak_accelerations[key] = {}

    #         peak_accelerations[key]['unsuccessful'] = {
    #             'values': np.mean(peak_accelerations_unsuccessful),
    #             'times': np.mean(peak_times_unsuccessful)
    #         }

    peak_accelerations = {}
    sem_values = {}  # Dictionary to store SEM values for each angle

    # Assume acceleration_times array and corresponding acceleration data arrays are correctly synchronized
    zero_time_index = next(i for i, t in enumerate(acceleration_times) if t >= 0)  # Find index where time is zero or the first positive time

    # Calculate and store peak accelerations and their times for each bin and each outcome category (successful, unsuccessful)
    for outcome_category, data in [('success', success_data['data']['accelerations']), 
                                ('unsuccessful', unsuccessful_data['data']['accelerations'])]:
        for key, array_stack in data.items():
            if array_stack.shape[1] > zero_time_index:  # Ensure there is data after time zero
                peak_indices = np.argmax(array_stack[:, zero_time_index:], axis=1) + zero_time_index
                peak_values = array_stack[np.arange(array_stack.shape[0]), peak_indices]
                peak_times = np.array(acceleration_times)[peak_indices]

                # Initialize dictionary for key if not already present
                if key not in peak_accelerations:
                    peak_accelerations[key] = {}
                if key not in sem_values:
                    sem_values[key] = {}

                # Store mean peak values and times
                peak_accelerations[key][outcome_category] = {
                    'values': np.mean(peak_values),
                    'times': np.mean(peak_times)
                }

                # Calculate and store SEM for values and times
                value_sem = np.std(peak_values, ddof=1) / np.sqrt(len(peak_values))  # ddof=1 for sample standard deviation
                time_sem = np.std(peak_times, ddof=1) / np.sqrt(len(peak_times))
                sem_values[key][outcome_category] = {
                    'value_sem': value_sem,
                    'time_sem': time_sem
                }

    # Optionally, store also the standard error of the mean for peak accelerations and times
    for key in peak_accelerations:
        if 'success' in peak_accelerations[key]:
            successful_values = [array_stack[i, np.argmax(array_stack[i, zero_time_index:] + zero_time_index)] for i in range(array_stack.shape[0])]
            successful_times = [acceleration_times[np.argmax(array_stack[i, zero_time_index:] + zero_time_index)] for i in range(array_stack.shape[0])]
            peak_accelerations[key]['success']['value_sem'] = np.std(successful_values) / np.sqrt(len(successful_values))
            peak_accelerations[key]['success']['time_sem'] = np.std(successful_times) / np.sqrt(len(successful_times))

        if 'unsuccessful' in peak_accelerations[key]:
            unsuccessful_values = [array_stack[i, np.argmax(array_stack[i, zero_time_index:] + zero_time_index)] for i in range(array_stack.shape[0])]
            unsuccessful_times = [acceleration_times[np.argmax(array_stack[i, zero_time_index:] + zero_time_index)] for i in range(array_stack.shape[0])]
            peak_accelerations[key]['unsuccessful']['value_sem'] = np.std(unsuccessful_values) / np.sqrt(len(unsuccessful_values))
            peak_accelerations[key]['unsuccessful']['time_sem'] = np.std(unsuccessful_times) / np.sqrt(len(unsuccessful_times))



    # Initialize a list to store the maximum y-values
    max_values = []

    # Loop through each angle bin to find the maximum y-value including the SEM
    for angle in success_data['mean']['speeds'].keys():
        # Calculate potential maximum for this bin (success and unsuccessful)
        max_success = np.max(success_data['mean']['speeds'][angle])
        max_unsuccessful = np.max(unsuccessful_data['mean']['speeds'][angle])
        
        # Store the larger of the two maximums for each angle
        max_values.append(max(max_success, max_unsuccessful))

    # Compute the average of all maximum values across the keys
    average_max_speed = np.mean(max_values)

    # Loop through each angle bin to find the maximum y-value including the SEM
    for angle in success_data['mean']['accelerations'].keys():
        # Calculate potential maximum for this bin (success and unsuccessful)
        max_success = np.max(success_data['mean']['accelerations'][angle])
        max_unsuccessful = np.max(unsuccessful_data['mean']['accelerations'][angle])
        
        # Store the larger of the two maximums for each angle
        max_values.append(max(max_success, max_unsuccessful))

    # Compute the average of all maximum values across the keys
    average_max_acceleration = np.mean(max_values)

    

    average_max_speed *= 1.1
    average_max_acceleration *= 1.1

    if start_angle == 'all':
        sorted_angles = sorted(success_data['mean']['speeds'].keys())  # Sorting keys, assuming they are the same for all data types

        num_bins = len(success_data['mean']['speeds'])
        # Determine grid dimensions
        num_columns = int(np.ceil(np.sqrt(num_bins)))
        num_rows = int(np.ceil(num_bins / num_columns))
        for data_type in ['speeds', 'accelerations']:
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10), sharex=True, sharey=True)
            axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

            for i, angle in enumerate(sorted_angles):
                ax = axes[i]
                # global speed_times
                # global acceleration_times
                if data_type == 'speeds':
                    x = [round(time, 2) for time in speed_times]
                    ax.set_ylim(0, average_max_speed)  # Optional: set dynamic limits
                elif data_type == 'accelerations':
                    x = [round(time, 2) for time in acceleration_times]
                    ax.set_ylim(-average_max_acceleration, average_max_acceleration)  # Optional: set dynamic limits
                # Plotting the mean data for successful and unsuccessful trials
                ax.plot(x, success_data['mean'][data_type][angle], label='Successful trials', color='blue')
                # ax.plot(x, unsuccessful_data['data'][data_type][angle], label='Unsuccessful trials', color='red')
                # draw a vertical line at 0:
                ax.axvline(x=0, color='black', linestyle='--')
                # Shading the SEM around each line
                ax.fill_between(x, 
                                success_data['mean'][data_type][angle] - success_data['sem'][data_type][angle], 
                                success_data['mean'][data_type][angle] + success_data['sem'][data_type][angle], 
                                color='blue', alpha=0.3)
                # ax.fill_between(x, 
                #                 unsuccessful_data['data'][data_type][angle] - unsuccessful_data['sem'][data_type][angle], 
                #                 unsuccessful_data['data'][data_type][angle] + unsuccessful_data['sem'][data_type][angle], 
                #                 color='red', alpha=0.3)

                
                if i >= 8:  # Set x-axis labels for the bottom row
                    ax.set_xticks(x)
                    ax.set_xticklabels(x, rotation=45)  # Rotate labels if necessary

            # Turn off unused axes
            for j in range(i + 1, num_rows * num_columns):
                axes[j].axis('off')

            # Adding common axis labels and title
            fig.text(0.5, -0.005, 'Time (seconds)', ha='center', va='center', fontsize=18)
            if data_type == 'speeds':
                y_text = 'Speed (pixels/sec)'
            elif data_type == 'accelerations':
                y_text = 'Acceleration (pixels/sec^2)'
            fig.text(-0.006, 0.5, y_text, ha='center', va='center', rotation='vertical', fontsize=18)
            fig.suptitle(f"{title} ({data_type})", fontsize=20, x=0.5, y=0.93)  # Replace 'Title Here' with your actual title

            # Adding a legend and adjusting layout
            fig.legend(labels=['Successful trials', 'T -0'], 
                    loc='upper right', 
                    bbox_to_anchor=(1, 0.95))
            plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust as needed
            plt.show()
    
        sorted_angles = sorted(peak_accelerations.keys())

        # Create a plot for peak acceleration times alongside the existing plots for speeds and accelerations
        fig, ax = plt.subplots(1, figsize=(12, 6))

        peak_times_success = []
        peak_times_unsuccessful = []
        sem_times_success = []
        sem_times_unsuccessful = []

        # Extracting peak times and SEMs for each angle
        for angle in sorted_angles:
            if 'success' in peak_accelerations[angle]:
                peak_times_success.append(peak_accelerations[angle]['success']['times'])
                sem_times_success.append(sem_values[angle]['success']['time_sem'])
            if 'unsuccessful' in peak_accelerations[angle]:
                peak_times_unsuccessful.append(peak_accelerations[angle]['unsuccessful']['times'])
                sem_times_unsuccessful.append(sem_values[angle]['unsuccessful']['time_sem'])

        # Plot peak times for successful and unsuccessful trials
        ax.plot(sorted_angles, peak_times_success, label='Successful trials', marker='o', linestyle='-', color='blue')
        # ax.plot(sorted_angles, peak_times_unsuccessful, label='Unsuccessful trials', marker='x', linestyle='-', color='red')
        # Adding shaded areas for SEM
        ax.fill_between(sorted_angles, 
                        np.array(peak_times_success) - np.array(sem_times_success), 
                        np.array(peak_times_success) + np.array(sem_times_success), 
                        color='blue', alpha=0.3)

        # ax.fill_between(sorted_angles, 
        #                 np.array(peak_times_unsuccessful) - np.array(sem_times_unsuccessful), 
        #                 np.array(peak_times_unsuccessful) + np.array(sem_times_unsuccessful), 
        #                 color='red', alpha=0.3)

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Time of Peak Acceleration (seconds)')
        ax.set_title('Time of Peak Acceleration vs Angle')
        ax.legend()
        plt.grid(True)
        plt.show()