import matplotlib.pyplot as plt
import numpy as np
import math

def error_trial_port_selection(sessions, 
                             title='title', 
                             start_angle='all', 
                             plot_type='line',
                             angle_mode='normal'):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot
    - plot_type: default: 'line', or 'radial'
    - angle_mode: default: 'normal', or 'delta'

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
                        bin['timeout'].append(trial)
        
        # now ech mouse has its trials sorted into bins based on angle, and success.

    # for each trial in successful trials, get the correct port, get the dlc data,
    # get cue offset time, and then grab the head angle at that time.
    # then use the calibrated angles to get the angle that mouse was facing relative to the correct port.
    # Append this to the correct bin.

    num_bins = 24
    bin_size = round(360 / num_bins)

    def get_data(trials, mouse_id):
        
        bins = {i: [] for i in range(-180, 180, bin_size)}
        # print(mice[mouse_id]['rig_id'])
        for trial in trials:
            correct_port = int(trial['correct_port'][-1]) - 1
            dlc_data = trial['DLC_data']
            timestamps = dlc_data['timestamps']
            cue_offset_time = trial['cue_end']
            sensor_touch_time = trial['next_sensor']['sensor_start']
            start_angle = abs(trial['turn_data']['cue_presentation_angle'])

            if mice[mouse_id]['rig_id'] == 1:
                port_angles = [64, 124, 184, 244, 304, 364] 
                # port_angles = [140, 200, 260, 320, 380, 440]
            elif mice[mouse_id]['rig_id'] == 2:
                port_angles = [240, 300, 360, 420, 480, 540]
                # port_angles = [140, 200, 260, 320, 380, 440]
            else:
                raise ValueError(f"Invalid rig ID: {mice[mouse_id]['rig_id']}")

            # get timestamps from dlc data:
            index = np.searchsorted(timestamps, sensor_touch_time, side='left') - 1

            cue_offset_coords = dlc_data.iloc[index]

            # to get mouse heading, take the ear coords, find angle between that and the nose coords, and then add 90 degrees to that angle.
            left_ear_coords = (cue_offset_coords["left_ear"]["x"], cue_offset_coords["left_ear"]["y"])
            left_likelihood = cue_offset_coords["left_ear"]["likelihood"]

            right_ear_coords = (cue_offset_coords["right_ear"]["x"], cue_offset_coords["right_ear"]["y"])
            right_likelihood = cue_offset_coords["right_ear"]["likelihood"]

            average_likelihood = (left_likelihood + right_likelihood) / 2
            # print(average_likelihood)

            vector_x = right_ear_coords[0] - left_ear_coords[0]
            vector_y = right_ear_coords[1] - left_ear_coords[1]

            # Calculate the angle relative to the positive x-axis
            theta_rad = math.atan2(-vector_y, vector_x)
            theta_deg = math.degrees(theta_rad)
            theta_deg = (theta_deg + 90) % 360

            # Calculating the midpoint
            midpoint_x = (left_ear_coords[0] + right_ear_coords[0]) / 2
            midpoint_y = (left_ear_coords[1] + right_ear_coords[1]) / 2

            # Midpoint coordinates
            midpoint = (midpoint_x, midpoint_y)

            if correct_port == "audio-1":
                correct_port = 1
            port = int(correct_port)
            cue_angle = port_angles[port]

            cue_presentation_angle = (cue_angle - theta_deg) % 360

            if cue_presentation_angle > 180:
                cue_presentation_angle -= 360
            elif cue_presentation_angle <= -180:
                cue_presentation_angle += 360

            end_angle = abs(cue_presentation_angle)

            delta_angle = start_angle - end_angle
            
            delta_angle = delta_angle % 360
            if delta_angle > 180:
                delta_angle -= 360

            for bin in bins:
                if delta_angle < bin + bin_size and delta_angle >= bin:
                    bins[bin].append(1)


        
        bin_lengths = [len(bins[bin]) for bin in bins]
        bin_lengths = np.array(bin_lengths, dtype=int)
    
        return bin_lengths

        # print(bin_lengths)
        
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

    bin_titles = [key for key in success_data['data']]
    if plot_type == 'line':
        if start_angle == 'all':
        
            fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
            axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

            bin_titles = [key for key in success_data['data'].keys()]

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
            fig.text(0.5, -0.005, 'Angle to correct port at cue offset', ha='center', va='center', fontsize=18)
            fig.text(-0.006, 0.5, 'Average number of trials', ha='center', va='center', rotation='vertical', fontsize=18)
            fig.suptitle(title, fontsize=20, x=0.5, y=0.93)  # Replace 'Title Here' with your actual title

            # Adding a legend and adjusting layout
            fig.legend(labels=['Successful trials', 'Unsuccessful trials'], 
                    loc='upper right', 
                    bbox_to_anchor=(1, 0.95),
                    )
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
            
    if plot_type == 'radial':

        fig, axes = plt.subplots(2, 6, subplot_kw={'projection': 'polar'}, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to make indexing easier

        for i, angle in enumerate(success_data['data'].keys()):
            ax = axes[i]
            theta = np.linspace(-np.pi, np.pi, len(success_data['data'][angle]))  # Convert linear space to radians from 0 to π

            # Convert 'angle' from degrees to radians for plotting
            ax.plot(theta, success_data['data'][angle], label='Successful trials', color='blue')
            ax.plot(theta, unsuccessful_data['data'][angle], label='Unsuccessful trials', color='red')

            # Fill between for SEM
            ax.fill_between(theta,
                            success_data['data'][angle] - success_data['sem'][angle],
                            success_data['data'][angle] + success_data['sem'][angle],
                            color='blue', alpha=0.3)
            ax.fill_between(theta,
                            unsuccessful_data['data'][angle] - unsuccessful_data['sem'][angle],
                            unsuccessful_data['data'][angle] + unsuccessful_data['sem'][angle],
                            color='red', alpha=0.3)

            ax.set_theta_zero_location('N')  # 0 degrees at the top
            ax.set_theta_direction(-1)  # Clockwise direction

            ax.set_title(f"Cue angle: {angle} degrees")
            ax.set_ylim(0, global_max)  # Optional: set dynamic limits

            n_legend = (f"n_success = {n_numbers['success'][angle]['mean']:.2f} ± {n_numbers['success'][angle]['sem']:.2f}\n"
                        f"n_fail = {n_numbers['fail'][angle]['mean']:.2f} ± {n_numbers['fail'][angle]['sem']:.2f}\n"
                        f"n_timeout = {n_numbers['timeout'][angle]['mean']:.2f} ± {n_numbers['timeout'][angle]['sem']:.2f}")
            ax.text(0.5, 0, n_legend, ha='center', va='top', transform=ax.transAxes, fontsize=9)
            # Limit the display of angles to 180 degrees
            # ax.set_xticks(np.pi / 180 * np.array([0, 30, 60, 90, 120, 150, 180]))  # Only show labels to 180 degrees
            # ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
            # ax.set_xticks([0])
            ax.set_xticklabels(['0°'])

        # Adding common labels and titles
        fig.text(0.5, 0.04, 'Angle to correct port at cue offset', ha='center', va='center', fontsize=18)
        fig.text(0.04, 0.5, 'Average number of trials', ha='center', va='center', rotation='vertical', fontsize=18)
        fig.suptitle(title, fontsize=20, x=0.5, y=0.98)

        # Legend and layout adjustment
        fig.legend(['Successful trials', 'Unsuccessful trials'], loc='upper right', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust as needed
