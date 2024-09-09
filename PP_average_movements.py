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
                    'sem': {}, 
                    'sd': {}}
    unsuccessful_data = {'data': {}, 
                         'sem': {}, 
                         'sd': {}}
    
    n_numbers = {'success': {}, 'fail': {}, 'timeout': {}}

    # for each mouse:
    for i, mouse in enumerate(mice):
        # for each angle bin, get the array of heading angles at cue offset time.
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
        combined_df = pd.concat(trajectory_list)
        success_data['data'][key] = combined_df.groupby(combined_df.index).mean()
        
    for key, trajectory_list in unsuccessful_data['data'].items():
        combined_df = pd.concat(trajectory_list)
        unsuccessful_data['data'][key] = combined_df.groupby(combined_df.index).mean()

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
    print('Done')
    
    from matplotlib import cm
    # Assuming your figure and axes setup remains the same
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    # Define color maps for successful and unsuccessful trials
    cmap_success = cm.winter
    cmap_unsuccess = cm.Wistia

    for i, key in enumerate(sorted(success_data['data'].keys())):
        ax = axes[i]

        # Set axis limits and aspect ratio
        # ax.set_xlim(0.4, 0.7)
        # ax.set_ylim(0.2, 0.8)
        # ax.set_xlim(0.4, 0.7)
        ax.set_xlim(0, 1)

        ax.set_ylim(0, 1)
        ax.set_aspect(1/1)

        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot successful trials with color gradient using arrows
        if key in success_data['data']:
            df_success = success_data['data'][key]
            # print(df_success['angle'])
            for j in range(len(df_success)):
                angle = df_success['angle'].iloc[j]
                # based on the position of x, find dx by the angle:
                dx = 0.1 * np.cos(np.radians(angle))
                dy = 0.1 * np.sin(np.radians(angle))
                # dx = df_success['new_x'].iloc[j] - df_success['x'].iloc[j]
                # dy = df_success['new_y'].iloc[j] - df_success['y'].iloc[j]
                color = cmap_success(j / len(df_success))
                ax.arrow(df_success['x'].iloc[j], df_success['y'].iloc[j], dx, dy,
                        color=color, width=0.005, head_width=0.02, head_length=0.015, length_includes_head=True)

        # # Plot unsuccessful trials with a different color gradient using arrows
        # if key in unsuccessful_data['data']:
        #     df_unsuccessful = unsuccessful_data['data'][key]
        #     for j in range(len(df_unsuccessful)):
        #         dx = df_unsuccessful['new_x'].iloc[j] - df_unsuccessful['x'].iloc[j]
        #         dy = df_unsuccessful['new_y'].iloc[j] - df_unsuccessful['y'].iloc[j]
        #         color = cmap_unsuccess(j / len(df_unsuccessful))
        #         ax.arrow(df_unsuccessful['x'].iloc[j], df_unsuccessful['y'].iloc[j], dx, dy,
        #                 color=color, head_width=0.01, head_length=0.015, length_includes_head=True)

        # Customizing each subplot with clear title and legend
        ax.set_title(f"Bin {key}°")
        n_legend = (f"n_success = {n_numbers['success'][key]['mean']:.2f} ± {n_numbers['success'][key]['sem']:.2f}\n"
                    f"n_fail = {n_numbers['fail'][key]['mean']:.2f} ± {n_numbers['fail'][key]['sem']:.2f}\n"
                    f"n_timeout = {n_numbers['timeout'][key]['mean']:.2f} ± {n_numbers['timeout'][key]['sem']:.2f}")
        ax.text(0.02, 0.98, n_legend, ha='left', va='top', transform=ax.transAxes, fontsize=9)
        # custom_lines = [plt.Line2D([0], [0], color=cmap_success(0.8), lw=4),
        #                 plt.Line2D([0], [0], color=cmap_unsuccess(0.8), lw=4)]
        
        custom_lines = [plt.Line2D([0], [0], color=cmap_success(0.8), lw=4)
                        ]
        ax.legend(custom_lines, ['Successful Trials', 'Unsuccessful Trials'], loc='lower left')

    plt.tight_layout()