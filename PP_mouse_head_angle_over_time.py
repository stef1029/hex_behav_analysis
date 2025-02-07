import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d

def plot_average_movements(sessions, 
                        title='title', 
                        start_angle='all'):
    """
    ### Inputs:
    - sessions: list of session objects
    - title: title of the plot
    - start_angle: default: 'all', or a specific angle to plot

    This function takes the list of sessions to use and extracts the trials per mouse.
    Split trials into successful and unsuccessful trials.
    Using cue presentation angle, bin the trials into angle groups.
    The, for each trial, get the DLC coords, interpolate them to fit a set length, 
    align them to the midpoint, and rotate the coordinates to zero them based on the angle they were binned by.
    Then plot these points as x and y to show the average movement of the mouse at that angle.
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

    new_length = 30
    framerate = 30

    def get_data(trials, mouse_id):

        num_frames = 30
        min_frames = 2
        trajectories = []

        # print(mice[mouse_id]['rig_id'])
        for trial in trials:

            df = trial['DLC_data']
            cue_presentation_angle = trial['turn_data']['cue_presentation_angle']
            flip = 0 if cue_presentation_angle > 0 else 1
            df.reset_index(drop=True, inplace=True)
            # print(df.head())
            if len(df) >= min_frames:

                max_x, max_y = 1280, 1024

                df[('new_left_ear', 'x')] = df[('left_ear', 'x')]
                df[('new_left_ear', 'y')] = df[('left_ear', 'y')]
                df[('new_right_ear', 'x')] = df[('right_ear', 'x')]
                df[('new_right_ear', 'y')] = df[('right_ear', 'y')]

                df['midpoint_x'] = (df[('left_ear', 'x')] + df[('right_ear', 'x')]) / 2
                df['midpoint_y'] = (df[('left_ear', 'y')] + df[('right_ear', 'y')]) / 2

                vec_x = df[('new_right_ear', 'x')] - df[('new_left_ear', 'x')]
                vec_y = df[('new_right_ear', 'y')] - df[('new_left_ear', 'y')]
                # Calculate angle and adjust by 90 degrees
                df['head_angle'] = np.arctan2(-vec_y, vec_x)

                # Normalize and scale coordinates to 0-1
                df['midpoint_x'] /= max_x
                df['midpoint_y'] /= max_y
                df[('new_left_ear', 'x')] /= max_x
                df[('new_left_ear', 'y')] /= max_y
                df[('new_right_ear', 'x')] /= max_x
                df[('new_right_ear', 'y')] /= max_y

                if flip:
                    df['head_angle'] = (2 * np.pi) - df['head_angle']
                    df['midpoint_x'] = 1 - df['midpoint_x']
                    df[('new_left_ear', 'x')] = 1 - df[('new_left_ear', 'x')]
                    df[('new_right_ear', 'x')] = 1 - df[('new_right_ear', 'x')]
                
                # Step 3: Align trajectory to start at (0.5, 0.5) and initial direction facing up
                # Translation to start at (0.5, 0.5)
                x_offset = 0.5 - float(df.loc[0, 'midpoint_x'])
                y_offset = 0.5 - float(df.loc[0, 'midpoint_y'])

                df['midpoint_x'] += x_offset
                df['midpoint_y'] += y_offset
                df[(('new_left_ear', 'x'))] += x_offset
                df[(('new_left_ear', 'y'))] += y_offset
                df[(('new_right_ear', 'x'))] += x_offset
                df[(('new_right_ear', 'y'))] += y_offset
                
                # Rotate to make the initial direction face up
                initial_angle = float(df.loc[0, 'head_angle'])
                rotation_angle = initial_angle + np.pi # Adjust to face upward
                
                def rotate(origin, point, angle):
                    """
                    Rotate a point counterclockwise by a given angle around a given origin.

                    The angle should be given in radians.
                    """
                    ox, oy = origin
                    px, py = point

                    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
                    return qx, qy

                df['midpoint_x'], df['midpoint_y'] = rotate((0.5, 0.5), (df['midpoint_x'], df['midpoint_y']), rotation_angle)
                df[('new_left_ear', 'x')], df[('new_left_ear', 'y')] = rotate((0.5, 0.5), (df[('new_left_ear', 'x')], df[('new_left_ear', 'y')]), rotation_angle)
                df[('new_right_ear', 'x')], df[('new_right_ear', 'y')] = rotate((0.5, 0.5), (df[('new_right_ear', 'x')], df[('new_right_ear', 'y')]), rotation_angle)

                vec_x = df[('new_right_ear', 'x')] - df[('new_left_ear', 'x')]
                vec_y = df[('new_right_ear', 'y')] - df[('new_left_ear', 'y')]

                df['head_angle'] = np.arctan2(-vec_y, vec_x)
                df['head_angle'] = df['head_angle'] - df['head_angle'][0]
                df['head_angle'] = 2*np.pi - (df['head_angle'] - (np.pi / 2))

                df['sin_angle'] = np.sin(df['head_angle'])
                df['cos_angle'] = np.cos(df['head_angle'])

                # Interpolate midpoints and angles
                scaling_factor = len(df) / new_length
                interpolator_x = interp1d(np.linspace(0, 1, len(df)), df['midpoint_x'], kind='linear')
                interpolator_y = interp1d(np.linspace(0, 1, len(df)), df['midpoint_y'], kind='linear')
                # interpolator_angle = interp1d(np.linspace(0, 1, len(df)), df['head_angle'], kind='linear')
                interpolator_sin = interp1d(np.linspace(0, 1, len(df)), df['sin_angle'], kind='linear')
                interpolator_cos = interp1d(np.linspace(0, 1, len(df)), df['cos_angle'], kind='linear')


                new_index = np.linspace(0, 1, new_length) 
                interpolated_x = interpolator_x(new_index)
                interpolated_y = interpolator_y(new_index)
                # interpolated_angle = interpolator_angle(new_index)
                interpolated_sin = interpolator_sin(new_index)
                interpolated_cos = interpolator_cos(new_index)

                interpolated_angle_rad = np.arctan2(interpolated_sin, interpolated_cos)
                interpolated_angle = np.degrees(interpolated_angle_rad)

                # Calculate new points 50 pixels away in the direction of the head angle
                new_x = interpolated_x + 0.4 * np.cos(interpolated_angle)
                new_y = interpolated_y + 0.4 * np.sin(interpolated_angle)

                # Create the final DataFrame with interpolated trajectories and new points
                interpolated_trajectory = pd.DataFrame({
                    'x': interpolated_x,
                    'y': interpolated_y,
                    'angle': interpolated_angle,
                    'scale': scaling_factor
                })
                trajectories.append(interpolated_trajectory)

        if len(trajectories) > 0:
            # Calculate the average trajectory
            # Concatenate all DataFrames vertically (stacking similar rows on top of each other)
            combined_df = pd.concat(trajectories)

            # Group by the index and calculate the mean of each group
            # Since the index represents time steps, this will average across all trajectories at each time step
            average_trajectory = combined_df.groupby(combined_df.index).mean()

            # Resultant DataFrame has the averaged coordinates of all trajectories at each time step
   
            return average_trajectory
        else:
            return None
        
    n = len(mice)  # Number of empty arrays you want
    successful_trials_data = np.empty(n, dtype=object)
    unsuccessful_trials_data = np.empty(n, dtype=object)

    for i in range(n):
        successful_trials_data[i] = np.array([], dtype=int)
        unsuccessful_trials_data[i] = np.array([], dtype=int)


    success_data = {'data': {},
                    'mean': {}, 
                    'sem': {}, 
                    'sd': {},
                    'scale': {}}
    unsuccessful_data = {'data': {},
                         'mean': {},
                         'sem': {}, 
                         'sd': {},
                         'scale': {}}
    
    n_numbers = {'success': {}, 'fail': {}, 'timeout': {}}

    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heding angles at cue offset time.
        for key, bin in mice[mouse]['binned_trials'].items():
            successful_trials = bin['successful_trials']
            unsuccessful_trials = bin['unsuccessful_trials']
            timeout_trials = bin['timeout']
            # print(f"Mouse: {mouse}, Bin: {key}°, Successful: {len(successful_trials)}")
            average_trajectory = get_data(successful_trials, mouse)
            if average_trajectory is not None:
                if key not in success_data['data']:
                    success_data['data'][key] = [average_trajectory]
                else:
                    success_data['data'][key].append(average_trajectory)

            # print(f"Mouse: {mouse}, Bin: {key}°, Unsuccessful: {len(unsuccessful_trials)}")
            average_trajectory = get_data(unsuccessful_trials, mouse)
            if average_trajectory is not None:
                if key not in unsuccessful_data['data']:
                    unsuccessful_data['data'][key] = [average_trajectory]
                else:
                    unsuccessful_data['data'][key].append(average_trajectory)


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


    for key, trajectory_list in success_data['data'].items():
        scales = np.array([trajectory['scale'][0] for trajectory in trajectory_list])
        if key not in success_data['scale']:
            success_data['scale'][key] = {'data': [], 'mean': [], 'sem': []}
        success_data['scale'][key]['data'] = scales
        success_data['scale'][key]['mean'] = np.mean(scales)
        success_data['scale'][key]['sem'] = np.std(scales) / np.sqrt(len(scales))
        combined_df = pd.concat(trajectory_list)
        success_data['mean'][key] = combined_df.groupby(combined_df.index).mean()
        success_data['sem'][key] = combined_df.groupby(combined_df.index).sem()
        
    for key, trajectory_list in unsuccessful_data['data'].items():
        scales = np.array([trajectory['scale'][0] for trajectory in trajectory_list])
        if key not in unsuccessful_data['scale']:
            unsuccessful_data['scale'][key] = {'data': [], 'mean': [], 'sem': []}
        unsuccessful_data['scale'][key]['data'] = scales
        unsuccessful_data['scale'][key]['mean'] = np.mean(scales)
        unsuccessful_data['scale'][key]['sem'] = np.std(scales) / np.sqrt(len(scales))
        combined_df = pd.concat(trajectory_list)
        unsuccessful_data['mean'][key] = combined_df.groupby(combined_df.index).mean()
        unsuccessful_data['sem'][key] = combined_df.groupby(combined_df.index).sem()

    for key, list in n_numbers['success'].items():
        n_numbers['success'][key] = {
                                     'mean': np.mean(list),
                                     'sem': np.std(list) / np.sqrt(len(list)),
                                     'sd': np.std(list)}
        n_numbers['fail'][key] = {'mean': np.mean(n_numbers['fail'][key]),
                                    'sem': np.std(n_numbers['fail'][key]) / np.sqrt(len(n_numbers['fail'][key])),
                                    'sd': np.std(n_numbers['fail'][key])}
        n_numbers['timeout'][key] = {'mean': np.mean(n_numbers['timeout'][key]),
                                    'sem': np.std(n_numbers['timeout'][key]) / np.sqrt(len(n_numbers['timeout'][key])),
                                    'sd': np.std(n_numbers['timeout'][key])}
        
    
    # each bin in success_data['data'] contains a stack of trajectories. 
    # Each of these has an angles array and a corresponding sem array.
    # Calculate a times array, and scale this with the corresponding scale factor from success_data['scales']
    
    # Setup the plot dimensions and subplots
    sorted_angles = sorted(success_data['data'].keys())
    num_bins = len(sorted_angles)

    if len(sorted_angles) > 1:
        bin_width = sorted_angles[1] - sorted_angles[0]
    else:
        bin_width = 180 / len(sorted_angles)  # default calculation if only one bin
    # Calculate mid-bin angles
    target_angles = 90 - np.array([angle + bin_width / 2 for angle in sorted_angles])
    # print(target_angles)

    num_columns = int(np.ceil(np.sqrt(num_bins)))
    num_rows = int(np.ceil(num_bins / num_columns))

    scaled_times = {}
    times = np.linspace(0, new_length / framerate, new_length)
    for key in sorted_angles:
        if key not in scaled_times:
            scaled_times[key] = []
        scale = success_data['scale'][key]['mean']
        scaled_times[key] = times * scale

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)

    axs = axs.flatten()  # Flatten the array of axes, if it's multidimensional

    # Plotting each angle's data
    for i, key in enumerate(sorted_angles):
        ax = axs[i]
        
        # Extract data for this particular angle
        x = scaled_times[key]  # Scaled time data
        y = success_data['mean'][key]['angle'] - target_angles[i] # Example mean trajectory component
        y_err = success_data['sem'][key]['angle']  # Example SEM of the trajectory component

        # Plot the data
        ax.plot(x, y, label='Success', color='blue')

        # Plot the SEM
        ax.fill_between(x, y - y_err, y + y_err, color='blue', alpha=0.2)

        ax.set_title(f'Angle: {angle}°')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Units/frame^2')
        ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top margin to accommodate the title

    # do same again but plot all lines on same plot:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plotting all angle data on the same plot
    colors = plt.cm.viridis(np.linspace(0, 1, num_bins))  # Generate colors from a colormap
    for i, key in enumerate(sorted_angles):
        # Extract data for this particular angle
        x = scaled_times[key]  # Scaled time data
        y = success_data['mean'][key]['angle'] - target_angles[i]  # Example mean trajectory component
        y_err = success_data['sem'][key]['angle']  # Example SEM of the trajectory component
        
        # Plot the data
        ax.plot(x, y, label=f'Angle: {key}°', color=colors[i])
        
        # Plot the SEM as a shaded area
        ax.fill_between(x, y - y_err, y + y_err, color=colors[i], alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Adjusted Angle (degrees)')
    ax.legend(title='Angles', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust right margin to fit the legend outside the plot
    plt.show()