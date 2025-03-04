import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d
import cv2 as cv
import os

def get_images(session):
    """ 
    This function takes a session and returns a group of folders for each angle bin. The images in each folder 
    represent the cue presentation angle for the mouse in that trail, and have the image rotated and drawn on to show the calculation. 
    ### Inputs:
    - session: a single session object that I want to get the images from.
    """

    mice = {}
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
    bin_size = round(360 / num_bins)

    # for each mouse, creates a binned trials dictionary, which contains all the trials where the mouse had the cue angle within the bin range.
    bins = {i: {'all_trials': [], 'successful_trials': [], 'unsuccessful_trials': [], 'timeout': []} for i in range(0, 360, bin_size)}

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

    session_video_path = session.session_video
    # cap = cv.VideoCapture(session_video_path)


    def save_images(trials, mouse_id, bin_key, session_video_path, success=True):
        if success:
            tag = 'successful'
        else:
            tag = 'unsuccessful'
        folder_path = os.path.join('output', f'mouse_{mouse_id}', f'bin_{bin_key}', tag)
        os.makedirs(folder_path, exist_ok=True)
        
        cap = cv.VideoCapture(str(session_video_path))
        # print(len(trials))
        for i, trial in enumerate(trials):
            start_frame = trial['video_frames'][0]
            mouse_heading = float(trial['turn_data']['bearing'])
            midpoint = np.array(trial['turn_data']['midpoint'], dtype=np.float32)
            port_position = np.array(trial['turn_data']['port_position'], dtype=np.float32)
            cue_angle = round(trial['turn_data']['cue_presentation_angle'])

            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                center = (int(w // 2), int(h // 2))  # Explicit casting to int

                # First rotation
                M = cv.getRotationMatrix2D(center, -mouse_heading + 90, 1.0)
                rotated = cv.warpAffine(frame, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))

                # Calculate new positions of the midpoint and port position after rotation
                new_midpoint = M[:, :2].dot(midpoint) + M[:, 2]
                new_port_position = M[:, :2].dot(port_position) + M[:, 2]

                # Calculate translation needed to center the midpoint
                translation_x = int(center[0] - new_midpoint[0])
                translation_y = int(center[1] - new_midpoint[1])

                # Creating a translation matrix
                M_trans = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
                translated = cv.warpAffine(rotated, M_trans, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))

                # Apply final translation to the midpoint and port position
                final_midpoint = new_midpoint + [translation_x, translation_y]
                # final_midpoint[1] -= 50
                final_port_position = new_port_position + [translation_x, translation_y]

                # Draw a vertical green line from the translated midpoint
                cv.line(translated, (int(final_midpoint[0]), 0), (int(final_midpoint[0]), h), (0, 255, 0), 2)

                # Draw a line between the translated midpoint and the port position
                cv.line(translated, (int(final_midpoint[0]), int(final_midpoint[1])), (int(final_port_position[0]), int(final_port_position[1])), (255, 0, 0), 2)

                # Draw cue angle text near the mouse's head
                text_position = (int(final_midpoint[0] + 10), int(final_midpoint[1] - 10))  # Adjust position relative to the midpoint
                cv.putText(translated, f'Cue Angle: {cue_angle}', text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Save the transformed image
                filename = os.path.join(folder_path, f'trial_{i}.jpg')
                cv.imwrite(filename, translated)
                # print(f"Saved transformed frame from start frame {start_frame} for trial {i} in {filename}")
            else:
                print(f"Failed to read frame at start frame {start_frame} for trial {i}")
        cap.release()

    # Usage example in the larger script:
    for key, bin in mice[mouse_id]['binned_trials'].items():
        successful_trials = bin['successful_trials']
        unsuccessful_trials = bin['unsuccessful_trials']
        save_images(successful_trials, mouse_id, key, session_video_path, success=True)
        save_images(unsuccessful_trials, mouse_id, key, session_video_path, success=False)

    # first_bin_key = None
    # for key in sorted(mice[mouse_id]['binned_trials']):
    #     if mice[mouse_id]['binned_trials'][key]['successful_trials']:
    #         first_bin_key = key
    #         break

    # if first_bin_key is not None:
    #     successful_trials = mice[mouse_id]['binned_trials'][first_bin_key]['successful_trials']
    #     save_images(successful_trials, mouse_id, first_bin_key, session_video_path)
    # else:
    #     print("No successful trials found in any bin.")