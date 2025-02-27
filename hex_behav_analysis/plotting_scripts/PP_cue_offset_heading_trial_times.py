import matplotlib.pyplot as plt
import numpy as np
import math

def mouse_heading_cue_offset(sessions, title='title', start_angle='all', failure_mode='all'):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot
    - failure_mode: default: 'all', 'timeouts', 'errors' or 'split' to see the timeouts and error touches seperately.

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
                if trial['DLC_data'] is not None and trial['cue_end'] is not None:
                    if trial['next_sensor'] != {}:
                        if trial['correct_port'] == trial['next_sensor']['sensor_touched']:
                            bin['successful_trials'].append(trial)
                        else:
                            bin['unsuccessful_trials'].append(trial)
                    else:
                        if failure_mode == 'split' or failure_mode == 'timeouts':
                            bin['timeout'].append(trial)
                        else:
                            bin['unsuccessful_trials'].append(trial)
        
        # now ech mouse has its trials sorted into bins based on angle, and success.

    # for each trial in successful trials, get the correct port, get the dlc data,
    # get cue offset time, and then grab the head angle at that time.
    # then use the calibrated angles to get the angle that mouse was facing relative to the correct port.
    # this function is called for every trial list in the angle bins. It should therefore return an 
    # array of points to plot, with the x being trial time and y being end angle

    def get_data(trials, mouse_id):
        
        trial_times = []
        end_angles = []

        # print(mice[mouse_id]['rig_id'])
        for trial in trials:
            correct_port = int(trial['correct_port'][-1]) - 1
            dlc_data = trial['DLC_data']
            timestamps = dlc_data['timestamps']
            cue_offset_time = trial['cue_end']
            trial_start = trial['cue_start']

            # get timestamps from dlc data:
            if cue_offset_time == None:
                print(trial)
            index = np.searchsorted(timestamps, cue_offset_time, side='left') - 1

            cue_offset_coords = dlc_data.iloc[index]

            # to get mouse heading, take the ear coords, find angle between that and the nose coords, and then add 90 degrees to that angle.
            left_ear_coords = (cue_offset_coords["left_ear"]["x"], cue_offset_coords["left_ear"]["y"])

            right_ear_coords = (cue_offset_coords["right_ear"]["x"], cue_offset_coords["right_ear"]["y"])

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

            # port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
            
            if mice[mouse_id]['rig_id'] == 1:
                port_angles = [64, 124, 184, 244, 304, 364] 
                # port_angles = [140, 200, 260, 320, 380, 440]
            elif mice[mouse_id]['rig_id'] == 2:
                port_angles = [240, 300, 360, 420, 480, 540]
                # port_angles = [140, 200, 260, 320, 380, 440]
            else:
                raise ValueError(f"Invalid rig ID: {mice[mouse_id]['rig_id']}")
            
            if correct_port == "audio-1":
                correct_port = 1
            port = int(correct_port)
            cue_angle = port_angles[port]

            cue_presentation_angle = (cue_angle - theta_deg) % 360

            if cue_presentation_angle > 180:
                cue_presentation_angle -= 360
            elif cue_presentation_angle <= -180:
                cue_presentation_angle += 360

            cue_presentation_angle = abs(cue_presentation_angle)

            # for each trial, return an array of the trial time and angle. 
            
            trial_times.append(trial_start)
            end_angles.append(cue_presentation_angle)


    
        return trial_times, end_angles


    success_data = {}
    unsuccessful_data = {}
    if failure_mode == 'split' or failure_mode == 'timeouts':
        timeout_data = {}
    
    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heading angles at cue offset time.
        for key, bin in mice[mouse]['binned_trials'].items():
            successful_trials = bin['successful_trials']
            unsuccessful_trials = bin['unsuccessful_trials']
            if failure_mode == 'split' or failure_mode == 'timeouts':
                timeout_trials = bin['timeout']

            trial_times, end_angles = get_data(successful_trials, mouse)
            if key not in success_data:
                success_data[key] = {'times': [], 'angles': []}
            success_data[key]['times'] += trial_times
            success_data[key]['angles'] += end_angles


            trial_times, end_angles = get_data(unsuccessful_trials, mouse)
            if key not in unsuccessful_data:
                unsuccessful_data[key] = {'times': [], 'angles': []}
            unsuccessful_data[key]['times'] += trial_times
            unsuccessful_data[key]['angles'] += end_angles

            if failure_mode == 'split' or failure_mode == 'timeouts':
                trial_times, end_angles = get_data(timeout_trials, mouse)
                if key not in timeout_data:
                    timeout_data[key] = {'times': [], 'angles': []}
                timeout_data[key]['times'] += trial_times
                timeout_data[key]['angles'] += end_angles



    if start_angle == 'all':
        # Create a grid of subplots with shared x and y axes
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

        for i, angle in enumerate(sorted(success_data.keys())):
            ax = axes[i]
            # Gather data for plotting
            success_times = success_data[angle]['times']
            success_angles = success_data[angle]['angles']
            unsuccessful_times = unsuccessful_data[angle]['times']
            unsuccessful_angles = unsuccessful_data[angle]['angles']
            if failure_mode == 'split' or failure_mode == 'timeouts':
                timeout_times = timeout_data[angle]['times']
                timeout_angles = timeout_data[angle]['angles']

            # Plotting successful and unsuccessful trials
            ax.scatter(success_times, success_angles, color='dodgerblue', label='Successful trials', alpha=0.6)
            if failure_mode == 'errors' or 'all':
                ax.scatter(unsuccessful_times, unsuccessful_angles, color='crimson', label='Unsuccessful trials', alpha=0.6)
            if failure_mode == 'split' or failure_mode == 'timeouts':
                ax.scatter(timeout_times, timeout_angles, color='lime', label='Timeout trials', alpha=0.6)

            ax.set_title(f"Initial cue angle: {angle}")
            
            # Only set x and y labels on the edge subplots if necessary
            if i % 4 == 0:  # first column
                ax.set_ylabel('End Angle')
            if i >= 8:  # last row
                ax.set_xlabel('Trial Time')

        # Set common figure labels and title
        fig.suptitle(title, fontsize=20, x=0.5, y=0.98)  # Position the title at the top of the figure
        fig.text(0.5, 0.04, 'Trial Time', ha='center', va='center', fontsize=18)
        fig.text(0.01, 0.5, 'End Angle to Correct Port', ha='center', va='center', rotation='vertical', fontsize=18)

        # Add a legend outside the top right corner of the figure
        fig.legend(labels=['Successful trials', 'Unsuccessful trials'], loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit everything properly
        plt.show()
    else:
        # Handle specific angle plotting
        if int(start_angle) in success_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(success_data[int(start_angle)]['times'], success_data[int(start_angle)]['angles'], color='blue', label='Successful trials')
            ax.scatter(unsuccessful_data[int(start_angle)]['times'], unsuccessful_data[int(start_angle)]['angles'], color='red', label='Unsuccessful trials')
            ax.set_title(f"Initial cue angle: {start_angle}")
            ax.set_xlabel('Trial Time (s)')
            ax.set_ylabel('End Angle to Correct Port')
            ax.legend()
            plt.show()
        else:
            raise ValueError(f"Invalid start angle: {start_angle}")