import matplotlib.pyplot as plt
import numpy as np
import math

def mouse_heading_cue_offset(sessions, 
                             title='title', 
                             start_angle='all', 
                             plot_type='line',
                             angle_mode='normal',
                             offset = 0):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot
    - plot_type: default: 'line', or 'radial'
    - angle_mode: default: 'normal', or 'delta'
    - offset: default: 0, or a value to add to the cue offset time to get the time to get the heading angle at.

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
        bins = {i: {'all_trials': [], 'successful_trials': [], 'unsuccessful_trials': []} for i in range(0, 180, bin_size)}

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
        
        # now ech mouse has its trials sorted into bins based on angle, and success.

    # for each trial in successful trials, get the correct port, get the dlc data,
    # get cue offset time, and then grab the head angle at that time.
    # then use the calibrated angles to get the angle that mouse was facing relative to the correct port.
    # Append this to the correct bin.

    num_bins = 12
    bin_size = round(180 / num_bins)

    def get_data(trials, mouse_id):
        
        bins = {i: [] for i in range(0, 180, bin_size)}
        # print(mice[mouse_id]['rig_id'])
        for trial in trials:
            correct_port = int(trial['correct_port'][-1]) - 1
            dlc_data = trial['DLC_data']
            timestamps = dlc_data['timestamps']
            cue_offset_time = trial['cue_end'] + offset
            start_angle = abs(trial['turn_data']['cue_presentation_angle'])

            # get timestamps from dlc data:
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

            # -------- GET CUE PRESENTATION ANGLE FROM MOUSE HEADING: ------------------------

            # port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
            if mice[mouse_id]['rig_id'] == 1:
                port_angles = [64, 124, 184, 244, 304, 364] 
            elif mice[mouse_id]['rig_id'] == 2:
                port_angles = [240, 300, 360, 420, 480, 540]
            else:
                raise ValueError(f"Invalid rig ID: {mice[mouse_id]['rig_id']}")

            # if the center is frame height/2, width/2, and the angle is the value in port_angles,
            # then the port coordinates are
            frame_height = 1080
            frame_width = 1280
            center_x = frame_width / 2
            center_y = frame_height / 2
            distance = (frame_height / 2) * 0.9 

            port_coordinates = []
            for angle_deg in port_angles:
                # Convert angle from degrees to radians
                angle_rad = np.deg2rad(angle_deg)
                
                # Calculate coordinates
                x = int(center_x + distance * np.cos(angle_rad))
                y = int(center_y - distance * np.sin(angle_rad))  # Subtracting to invert y-axis direction
                
                # Append tuple of (x, y) to the list of coordinates
                port_coordinates.append((x, y))

            relative_angles = []
            # Convert mouse heading to radians for calculation
            mouse_heading_rad = np.deg2rad(theta_deg)

            for port_x, port_y in port_coordinates:
                # Calculate vector from midpoint to the port
                vector_x = port_x - midpoint[0]
                vector_y = port_y - midpoint[1]

                # Calculate the angle from the x-axis to this vector
                port_angle_rad = math.atan2(-vector_y, vector_x)

                # Calculate the relative angle
                relative_angle_rad = port_angle_rad - mouse_heading_rad

                # Convert relative angle to degrees and make sure it is within [0, 360)
                relative_angle_deg = math.degrees(relative_angle_rad) % 360

                # Append calculated relative angle to list
                relative_angles.append(relative_angle_deg)

            correct_port = trial["correct_port"]
            if correct_port == "audio-1":
                correct_port = 1
            port = int(correct_port) - 1
            cue_presentation_angle = relative_angles[port] % 360

            if cue_presentation_angle > 180:
                cue_presentation_angle -= 360
            elif cue_presentation_angle <= -180:
                cue_presentation_angle += 360

            end_angle = abs(cue_presentation_angle)    #  Should probably come up with a way to also show > 180 degree angles on plot.

            delta_angle = start_angle - end_angle

            # bin this trial based on the cue_presentation_angle:
            if angle_mode == 'normal':
                for bin in bins:
                    if end_angle < bin + bin_size and end_angle >= bin:
                        bins[bin].append(1)
            elif angle_mode == 'delta':
                for bin in bins:
                    if delta_angle < bin + bin_size and delta_angle >= bin:
                        bins[bin].append(1)
            else:
                raise ValueError('angle_mode must be "normal" or "delta"')

        
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

    success_data = {'data': {}, 'sem': {}, 'sd': {}}
    unsuccessful_data = {'data': {}, 'sem': {}, 'sd': {}}
    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heding angles at cue offset time.
        for key, bin in mice[mouse]['binned_trials'].items():
            successful_trials = bin['successful_trials']
            unsuccessful_trials = bin['unsuccessful_trials']

            data = get_data(successful_trials, mouse)
            successful_trials_data[i] = data
            if key not in success_data['data']:
                success_data['data'][key] = data
            if key in success_data['data']:
                success_data['data'][key] = np.vstack((success_data['data'][key], data))

            data = get_data(unsuccessful_trials, mouse)
            unsuccessful_trials_data[i] = data
            if key not in unsuccessful_data['data']:
                unsuccessful_data['data'][key] = data
            if key in unsuccessful_data['data']:
                unsuccessful_data['data'][key] = np.vstack((unsuccessful_data['data'][key], data))

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

        fig, axes = plt.subplots(3, 4, subplot_kw={'projection': 'polar'}, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to make indexing easier

        for i, angle in enumerate(success_data['data'].keys()):
            ax = axes[i]
            theta = np.linspace(0, np.pi, len(success_data['data'][angle]))  # Convert linear space to radians from 0 to π

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

            ax.set_title(f"Initial cue angle: {angle} degrees")
            ax.set_ylim(0, max(np.max(success_data['data'][angle] + success_data['sem'][angle]), 
                            np.max(unsuccessful_data['data'][angle] + unsuccessful_data['sem'][angle])) * 1.1)  # Optional: set dynamic limits

            # Limit the display of angles to 180 degrees
            ax.set_xticks(np.pi / 180 * np.array([0, 30, 60, 90, 120, 150, 180]))  # Only show labels to 180 degrees
            ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])

        # Adding common labels and titles
        if angle_mode == 'normal':
            if offset == 0:
                fig.text(0.5, 0.02, 'Angle to correct port at cue offset', ha='center', va='center', fontsize=18)
            else:
                fig.text(0.5, 0.02, f'Angle to correct port at cue offset + {offset}s', ha='center', va='center', fontsize=18)
        if angle_mode == 'delta':
            if offset == 0:
                fig.text(0.5, 0.02, 'Mouse heading change at cue offset', ha='center', va='center', fontsize=18)
            else:
                fig.text(0.5, 0.02, f'Mouse heading change at cue offset + {offset}s', ha='center', va='center', fontsize=18)

        fig.text(0.04, 0.5, 'Average number of trials', ha='center', va='center', rotation='vertical', fontsize=18)
        fig.suptitle(title, fontsize=20, x=0.5, y=0.98)

        # Legend and layout adjustment
        fig.legend(['Successful trials', 'Unsuccessful trials'], loc='upper right', bbox_to_anchor=(0.8, 1))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust as needed
