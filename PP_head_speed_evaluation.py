import matplotlib.pyplot as plt
import numpy as np
import math

def evaluate_head_speed(sessions, 
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
        num_bins = 12
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
                    else:
                        bin['unsuccessful_trials'].append(trial)
                        bin['timeout'].append(trial)
        
        # now ech mouse has its trials sorted into bins based on angle, and success.

    # for each trial in successful trials, get the correct port, get the dlc data,
    # get cue offset time, and then grab the head angle at that time.
    # then use the calibrated angles to get the angle that mouse was facing relative to the correct port.
    # Append this to the correct bin.
    global bin_titles
    bin_titles = []

    def get_data(trials, mouse_id):

        max_bin = 1500
        num_bins = 10
        speeds = []
        timestamps = mice[mouse_id]['session'].video_timestamps

        # print(mice[mouse_id]['rig_id'])
        for trial in trials:
            correct_port = int(trial['correct_port'][-1]) - 1
            dlc_data = trial['DLC_data']
            cue_start = trial['cue_start']

            trial_start_index = np.searchsorted(timestamps, cue_start)

            # print(dlc_data)

            start_angle = abs(trial['turn_data']['cue_presentation_angle'])


            full_dlc_data = mice[mouse_id]['session'].DLC_coords

            num_frames = 5
            if trial_start_index - num_frames > 0:
                start_index = trial_start_index - num_frames
            else:
                start_index = 0

            dlc_data = full_dlc_data.iloc[start_index:trial_start_index]


            # Extract coordinates
            x1, y1 = dlc_data.iloc[0][('left_ear', 'x')], dlc_data.iloc[0][('left_ear', 'y')]
            x2, y2 = dlc_data.iloc[-1][('left_ear', 'x')], dlc_data.iloc[-1][('left_ear', 'y')]

            # Calculate distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Calculate total time elapsed
            total_time = (len(dlc_data) - 1) * 0.033  # Each row represents 1/30th of a second

            # Calculate speed
            speed = distance / total_time
            # print(speed)

            speeds.append(speed)

        bin_size = int(max_bin / num_bins)

        bins = {i: [] for i in range(0, max_bin, bin_size)}
        global bin_titles
        bin_titles = [i for i in range(0, max_bin, bin_size)]

        for speed in speeds:
            for bin in bins:
                if speed < bin + bin_size and speed >= bin:
                    bins[bin].append(speed)

        bin_lengths = [len(bins[bin]) for bin in bins]
        bin_lengths = np.array(bin_lengths, dtype=int)
    
        return bin_lengths
        
    n = len(mice)  # Number of empty arrays you want
    successful_trials_data = np.empty(n, dtype=object)
    unsuccessful_trials_data = np.empty(n, dtype=object)

    for i in range(n):
        successful_trials_data[i] = np.array([], dtype=int)
        unsuccessful_trials_data[i] = np.array([], dtype=int)


    success_data = {'data': {}, 
                    'sem': {}, 
                    'sd': {}}
    unsuccessful_data = {'data': {}, 
                         'sem': {}, 
                         'sd': {}}
    
    n_numbers = {'success': {}, 'fail': {}, 'timeout': {}}

    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heding angles at cue offset time.
        for key, bin in mice[mouse]['binned_trials'].items():
            successful_trials = bin['successful_trials']
            unsuccessful_trials = bin['unsuccessful_trials']
            timeout_trials = bin['timeout']

            data = get_data(successful_trials, mouse)
            if key not in success_data['data']:
                success_data['data'][key] = data
            if key in success_data['data']:
                success_data['data'][key] = np.vstack((success_data['data'][key], data))


            data = get_data(unsuccessful_trials, mouse)
            if key not in unsuccessful_data['data']:
                unsuccessful_data['data'][key] = data
            if key in unsuccessful_data['data']:
                unsuccessful_data['data'][key] = np.vstack((unsuccessful_data['data'][key], data))


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

                

    for key, array_stack in success_data['data'].items():
        # Sum along rows to combine all arrays into one
        success_data['data'][key] = np.mean(array_stack, axis=0)   
        success_data['sem'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        success_data['sd'][key] = np.std(array_stack, axis=0)


    for key, array_stack in unsuccessful_data['data'].items():
        # Sum along rows to combine all arrays into one
        unsuccessful_data['data'][key] = np.mean(array_stack, axis=0)  
        unsuccessful_data['sem'][key] = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
        unsuccessful_data['sd'][key] = np.std(array_stack, axis=0)

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

    # Initialize a variable to store the maximum y-value
    global_max = 0
    # Loop through each angle bin to find the maximum y-value including the SEM
    for angle in success_data['data'].keys():
        # Calculate potential maximum for this bin (success and unsuccessful)
        max_success = np.max(success_data['data'][angle] + success_data['sem'][angle])
        max_unsuccessful = np.max(unsuccessful_data['data'][angle] + unsuccessful_data['sem'][angle])

        # Update global_max if the current bin's max is higher
        if max_success > global_max:
            global_max = max_success
        if max_unsuccessful > global_max:
            global_max = max_unsuccessful

    # Scale the maximum up slightly to ensure data isn't touching the top of the plot
    global_max *= 1.1

    # bin_titles = [float(key) for key in success_data['data']]

    if start_angle == 'all':
    
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

        for i, angle in enumerate(success_data['data'].keys()):
            ax = axes[i]
            x = range(len(success_data['data'][angle]))  # Assuming x-axis corresponds to the length of the data array

            # Plotting the mean data for successful and unsuccessful trials
            ax.plot(x, success_data['data'][angle], label='Successful trials', color='blue')
            ax.plot(x, unsuccessful_data['data'][angle], label='Unsuccessful trials', color='red')

            # Shading the SEM around each line
            ax.fill_between(x, 
                            success_data['data'][angle] - success_data['sem'][angle], 
                            success_data['data'][angle] + success_data['sem'][angle], 
                            color='blue', alpha=0.3)
            ax.fill_between(x, 
                            unsuccessful_data['data'][angle] - unsuccessful_data['sem'][angle], 
                            unsuccessful_data['data'][angle] + unsuccessful_data['sem'][angle], 
                            color='red', alpha=0.3)

            ax.set_title(f"Initial cue angle: {angle}")
            if i >= 8:  # Set x-axis labels for the bottom row
                ax.set_xticks(x)
                ax.set_xticklabels(bin_titles, rotation=45)  # Rotate labels if necessary

        # Adding common axis labels and title
        fig.text(0.5, -0.005, 'Head speed at cue presentation (pixels/sec)', ha='center', va='center', fontsize=18)
        fig.text(-0.006, 0.5, 'Average number of trials', ha='center', va='center', rotation='vertical', fontsize=18)
        fig.suptitle(title, fontsize=20, x=0.5, y=0.93)  # Replace 'Title Here' with your actual title

        # Adding a legend and adjusting layout
        fig.legend(labels=['Successful trials', 'Unsuccessful trials'], 
                loc='upper right', 
                bbox_to_anchor=(1, 0.95))
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust as needed
        plt.show()
    
    else:
        if int(start_angle) in success_data['data']:
            start_angle = int(start_angle)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(success_data['data'][start_angle], label='Successful trials')
            ax.plot(unsuccessful_data['data'][start_angle], label='Unsuccessful trials')
            ax.set_title(f"Initial cue angle: {start_angle}")
            ax.set_xlabel('Angle to correct port at cue offset', fontsize=16)
            ax.set_ylabel('Average number of trials', fontsize=16)
            ax.set_xticklabels(bin_titles, rotation=45)
            ax.legend()
            ax.set_title(f"{title} - Initial cue angle: {start_angle}", fontsize=20)

            # fig2, ax2 = plt.subplots(figsize=(10, 6))
        else:
            raise ValueError(f"Invalid start angle: {start_angle}")
            
    # if plot_type == 'radial':

    #     fig, axes = plt.subplots(2, 6, subplot_kw={'projection': 'polar'}, figsize=(15, 10))
    #     axes = axes.flatten()  # Flatten to make indexing easier

    #     for i, angle in enumerate(success_data['data'].keys()):
    #         ax = axes[i]
    #         theta = np.linspace(-np.pi, np.pi, len(success_data['data'][angle]))  # Convert linear space to radians from 0 to π

    #         # Convert 'angle' from degrees to radians for plotting
    #         ax.plot(theta, success_data['data'][angle], label='Successful trials', color='blue')
    #         ax.plot(theta, unsuccessful_data['data'][angle], label='Unsuccessful trials', color='red')

    #         # Fill between for SEM
    #         ax.fill_between(theta,
    #                         success_data['data'][angle] - success_data['sem'][angle],
    #                         success_data['data'][angle] + success_data['sem'][angle],
    #                         color='blue', alpha=0.3)
    #         ax.fill_between(theta,
    #                         unsuccessful_data['data'][angle] - unsuccessful_data['sem'][angle],
    #                         unsuccessful_data['data'][angle] + unsuccessful_data['sem'][angle],
    #                         color='red', alpha=0.3)

    #         ax.set_theta_zero_location('N')  # 0 degrees at the top
    #         ax.set_theta_direction(-1)  # Clockwise direction

    #         ax.set_title(f"Cue angle: {angle} degrees")
    #         ax.set_ylim(0, global_max)  # Optional: set dynamic limits

    #         n_legend = (f"n_success = {n_numbers['success'][angle]['mean']:.2f} ± {n_numbers['success'][angle]['sem']:.2f}\n"
    #                     f"n_fail = {n_numbers['fail'][angle]['mean']:.2f} ± {n_numbers['fail'][angle]['sem']:.2f}\n"
    #                     f"n_timeout = {n_numbers['timeout'][angle]['mean']:.2f} ± {n_numbers['timeout'][angle]['sem']:.2f}")
    #         ax.text(0.5, 0, n_legend, ha='center', va='top', transform=ax.transAxes, fontsize=9)
    #         # Limit the display of angles to 180 degrees
    #         # ax.set_xticks(np.pi / 180 * np.array([0, 30, 60, 90, 120, 150, 180]))  # Only show labels to 180 degrees
    #         # ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
    #         # ax.set_xticks([0])
    #         ax.set_xticklabels(['0°'])

    #     # Adding common labels and titles
    #     fig.text(0.5, 0.04, 'Angle to correct port at cue offset', ha='center', va='center', fontsize=18)
    #     fig.text(0.04, 0.5, 'Average number of trials', ha='center', va='center', rotation='vertical', fontsize=18)
    #     fig.suptitle(title, fontsize=20, x=0.5, y=0.98)

    #     # Legend and layout adjustment
    #     fig.legend(['Successful trials', 'Unsuccessful trials'], loc='upper right', bbox_to_anchor=(1, 1))
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust as needed
