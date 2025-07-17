from pathlib import Path
import pandas as pd
import bisect
import cv2 as cv
import math
import numpy as np
import pickle
from pynwb import NWBHDF5IO, TimeSeries, ProcessingModule
import os
import traceback

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.DetectTrials import DetectTrials

class Session:
    def __init__(self, session_dict, recalculate=False, dlc_model_name=None):
        """
        Initialize a Session object.
        
        Args:
            session_dict (dict): Dictionary containing session information.
            recalculate (bool): If True, force recalculation of trials even if analysis pickle exists.
            dlc_model_name (str): Specific DLC model name to use. If None, uses default behaviour.
        """
        try:
            self.session_ID = session_dict.get('session_id')
        except Exception:
            raise ValueError("Invalid session dictionary, check that session ID matches what is in loaded cohort folder information.\
                             (Ranaming of files often causes this issue if all session IDs aren't updated in the rest of the code.)")
        try:
            self.session_dict = session_dict
            self.session_directory = Path(self.session_dict.get('directory'))
            self.dlc_model_name = dlc_model_name  # Store the model name
            print(f"Loading session {self.session_ID}...")
            
            self.portable = self.session_dict['portable']
            if self.portable:
                self.nwb_file_path = Path(self.session_dict.get('NWB_file'))
            else:
                self.nwb_file_path = Path(self.session_dict.get('processed_data', {}).get('NWB_file'))
                self.DLC_coords_path = Path(self.session_dict.get('processed_data', {}).get('DLC', {}).get('coords_csv'))

            # Define the analysis pickle file path (e.g. session123_analysis.pkl)
            self.analysis_pickle_path = self.nwb_file_path.parent / (self.nwb_file_path.stem + "_analysis.pkl")

            # Read required metadata from the original NWB file.
            with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
                nwbfile = io.read()
                self.last_timestamp = nwbfile.acquisition['scales'].timestamps[-1]
                session_metadata = nwbfile.experiment_description
                self.phase = str(session_metadata.split(";")[0].split(":")[1])
                self.rig_id = int(session_metadata.split(";")[1].split(":")[1])
            
            # Check if analysis pickle file exists and is non-empty.
            analysis_data = self._load_analysis_pickle()
            trials_present = "trials" in analysis_data and analysis_data["trials"]
            video_data_present = "video_data" in analysis_data and analysis_data["video_data"]

            self.load_data()

            if not trials_present or recalculate:
                # Process session to detect trials.
                self.trial_detector = DetectTrials(self.nwb_file_path, self.session_dict, self.rig_id)
                self.trials = self.trial_detector.trials
                self.save_trials_to_analysis()
                if recalculate:
                    print("Recalculating trials as requested...")
            else:
                self.load_trials_from_analysis()

            if not video_data_present or recalculate:
                self.add_video_data_to_trials()
                self.save_video_data_to_analysis()
            else:
                self.load_video_data_from_analysis()

            self.add_angle_data()

        except Exception:
            print(f"Error loading session {self.session_ID}")
            traceback.print_exc()
            pass

    def _save_analysis_pickle(self, analysis_dict):
        """Helper to dump analysis data to the pickle file."""
        with open(self.analysis_pickle_path, 'wb') as f:
            pickle.dump(analysis_dict, f)

    def _load_analysis_pickle(self):
        """Helper to load analysis data from the pickle file.
        Returns an empty dict if the file does not exist or is empty."""
        if not self.analysis_pickle_path.exists() or self.analysis_pickle_path.stat().st_size == 0:
            return {}
        with open(self.analysis_pickle_path, 'rb') as f:
            try:
                data = pickle.load(f)
                # Check if model name matches
                if self.dlc_model_name and data.get("dlc_model_name") != self.dlc_model_name:
                    print(f"Model mismatch in pickle: {data.get('dlc_model_name')} vs {self.dlc_model_name}. Will recalculate video data.")
                    # Remove video data to force recalculation
                    if "video_data" in data:
                        del data["video_data"]
                    if "video_timestamps" in data:
                        del data["video_timestamps"]
                return data
            except EOFError:
                return {}

    def save_trials_to_analysis(self):
        """
        Save detected trials into a pickle file.
        Stores a dictionary with key 'trials' and 'trials_timestamps' (from original NWB file).
        """
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            orig_nwb = io.read()
            ts = orig_nwb.acquisition['scales'].timestamps[:len(self.trials)]
        analysis_dict = self._load_analysis_pickle()
        analysis_dict["trials"] = self.trials
        analysis_dict["trials_timestamps"] = ts
        self._save_analysis_pickle(analysis_dict)
        # print("Trial data saved to analysis pickle.")

    def load_trials_from_analysis(self):
        """Load trial data from the analysis pickle file."""
        analysis_dict = self._load_analysis_pickle()
        if "trials" in analysis_dict:
            self.trials = analysis_dict["trials"]
            # Reset video-related fields so they can be updated later.
            for trial in self.trials:
                trial["video_frames"] = None
                trial["DLC_data"] = None
            # print("Trial data loaded from analysis pickle.")
        else:
            # print("No trial data found in analysis pickle.")
            pass

    def save_video_data_to_analysis(self):
        """
        Save video frame data and DLC data into the analysis pickle file.
        Stores a dictionary with key 'video_data' and 'video_timestamps'.
        """
        video_data = []
        for trial in self.trials:
            trial_video_data = {
                'trial_no': trial.get('trial_no'),
                'video_frames': trial.get('video_frames'),
                'DLC_data': trial.get('DLC_data')  # Pickle can serialize DataFrames as is.
            }
            video_data.append(trial_video_data)
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            orig_nwb = io.read()
            ts = orig_nwb.acquisition['scales'].timestamps[:len(video_data)]
        analysis_dict = self._load_analysis_pickle()
        analysis_dict["video_data"] = video_data
        analysis_dict["video_timestamps"] = ts
        analysis_dict["dlc_model_name"] = self.dlc_model_name  # Store model name
        self._save_analysis_pickle(analysis_dict)

    def load_video_data_from_analysis(self):
        """
        Load video data from the analysis pickle file and merge it into self.trials.
        """
        analysis_dict = self._load_analysis_pickle()
        if "video_data" in analysis_dict:
            video_data = analysis_dict["video_data"]
            for trial_video_data in video_data:
                trial_no = trial_video_data.get('trial_no')
                if trial_no is not None and trial_no < len(self.trials):
                    self.trials[trial_no]['video_frames'] = trial_video_data.get('video_frames')
                    self.trials[trial_no]['DLC_data'] = trial_video_data.get('DLC_data')
            # print("Video data loaded from analysis pickle.")
        else:
            # print("No video data found in analysis pickle.")
            pass

    def add_timestamps(self, trials):
        new_trials = []
        for trial in trials:
            new_trial = trial
            new_trial["start_time"] = self.timestamps[trial["start"]]
            try:
                new_trial["end_time"] = self.timestamps[trial["next_sensor_time"]]
            except (KeyError, TypeError):
                try:
                    new_trial["end_time"] = self.timestamps[trial['reward_touch']['end']]
                except:
                    new_trial["end_time"] = self.timestamps[-1]

            new_trials.append(new_trial)

        return new_trials

    def get_video(self, trial):
        """
        returns: (video object set to start frame, end frame)\n
        Should really be a function within a true trial object, but for now it's here.
        """
        # open video and grab frame:
        cap = cv.VideoCapture(str(self.session_video))
        cap.set(cv.CAP_PROP_POS_FRAMES, trial["video_frames"][0])

        return cap, trial["video_frames"][-1]
        

    def add_angle_data(self):
        for trial in self.trials:
            # Check DLC_data is non-empty and has enough rows for `buffer`
            if trial["DLC_data"] is not None and len(trial["DLC_data"]) >= 1:
                trial["turn_data"] = self.find_angles(trial)
            else:
                trial["turn_data"] = None



    def find_angles(self, trial, buffer=1):
        """
        Calculate angles for each trial with corrections for ear detection issues.
        
        This function computes the mouse's head bearing (direction it's facing) and the angle
        to the cued port relative to that bearing. It uses three different methods depending
        on the quality of the ear tracking:
        
        1. **Ear-based calculation** (default):
        - Uses positions of left and right ears to determine head orientation
        - Head direction is perpendicular to the line connecting the ears
        - Validates against spine data if available to detect flipped ears
        
        2. **Spine-based calculation** (when ears too close):
        - Triggered when ear distance < 50 pixels (likely both on same ear)
        - Fits a line through spine points (spine_1 to spine_4)
        - Uses line angle as body/head direction
        
        3. **Fallback ear calculation** (when spine data unavailable):
        - Uses ear positions even if too close together
        - May be inaccurate but prevents complete failure
        
        Coordinate system notes:
        - Video coordinates have y increasing downward (flipped from standard)
        - All atan2 calls use -y to account for this
        - Angles are in degrees, with 0° = rightward, 90° = upward
        
        Parameters
        ----------
        trial : dict
            Trial data containing DLC coordinates with body part positions.
            Must contain 'DLC_data' DataFrame with columns for each body part.
        buffer : int, default=1
            Number of frames to average over for position calculations.
            Helps reduce noise in tracking data.
            
        Returns
        -------
        dict or None
            Returns None if insufficient data.
            Otherwise returns dictionary containing:
            - bearing: float, mouse head angle in degrees (0-360)
            - midpoint: tuple, (x,y) position representing mouse head location
            - left_ear_likelihood: float, average DLC confidence for left ear
            - right_ear_likelihood: float, average DLC confidence for right ear
            - cue_presentation_angle: float, angle to cued port (-180 to 180)
            - port_touched_angle: float or None, angle to touched port if applicable
            - angle_correction_method: str, one of:
                - "none": standard ear-based calculation
                - "ear_flip_corrected": ears detected as flipped and corrected
                - "spine_based": used spine due to ears too close
                - "ears_too_close_no_spine": ears too close but no spine data
            - ear_distance: float, pixel distance between detected ear positions
            - spine_data_available: bool, whether spine tracking data was found
        """
        # Guard against empty or too-short DLC data
        if trial["DLC_data"] is None or len(trial["DLC_data"]) < buffer:
            return None
        
        # Gather ear coordinates and likelihoods over 'buffer' frames
        left_ear_coords = []
        left_ear_likelihoods = []
        right_ear_coords = []
        right_ear_likelihoods = []
        
        # Validate ear data before processing
        try:
            for i in range(buffer):
                left_x = trial["DLC_data"]["left_ear"]["x"].iloc[i]
                left_y = trial["DLC_data"]["left_ear"]["y"].iloc[i]
                right_x = trial["DLC_data"]["right_ear"]["x"].iloc[i]
                right_y = trial["DLC_data"]["right_ear"]["y"].iloc[i]
                
                # Check for NaN or infinite values
                if np.isfinite(left_x) and np.isfinite(left_y):
                    left_ear_coords.append((left_x, left_y))
                    left_ear_likelihoods.append(trial["DLC_data"]["left_ear"]["likelihood"].iloc[i])
                
                if np.isfinite(right_x) and np.isfinite(right_y):
                    right_ear_coords.append((right_x, right_y))
                    right_ear_likelihoods.append(trial["DLC_data"]["right_ear"]["likelihood"].iloc[i])
        except Exception as e:
            print(f"Error extracting ear data: {e}")
            return None
        
        # Ensure we have valid ear data
        if not left_ear_coords or not right_ear_coords:
            return None

        # Compute average coords and average likelihood for each ear
        average_left_ear = (
            sum(coord[0] for coord in left_ear_coords) / len(left_ear_coords),
            sum(coord[1] for coord in left_ear_coords) / len(left_ear_coords),
        )
        avg_left_ear_likelihood = sum(left_ear_likelihoods) / len(left_ear_likelihoods) if left_ear_likelihoods else 0

        average_right_ear = (
            sum(coord[0] for coord in right_ear_coords) / len(right_ear_coords),
            sum(coord[1] for coord in right_ear_coords) / len(right_ear_coords),
        )
        avg_right_ear_likelihood = sum(right_ear_likelihoods) / len(right_ear_likelihoods) if right_ear_likelihoods else 0

        # Check distance between ears
        ear_distance = math.sqrt(
            (average_right_ear[0] - average_left_ear[0])**2 + 
            (average_right_ear[1] - average_left_ear[1])**2
        )
        
        minimum_ear_distance = 50  # Minimum distance threshold
        angle_correction_method = "none"  # Track correction method used
        
        # Get spine data for validation/correction
        spine_positions = []
        spine_available = True
        
        try:
            for spine_idx in range(1, 5):  # spine_1 to spine_4
                spine_name = f"spine_{spine_idx}"
                if spine_name in trial["DLC_data"].columns.levels[0]:
                    spine_x_vals = []
                    spine_y_vals = []
                    for i in range(buffer):
                        x_val = trial["DLC_data"][spine_name]["x"].iloc[i]
                        y_val = trial["DLC_data"][spine_name]["y"].iloc[i]
                        if np.isfinite(x_val) and np.isfinite(y_val):
                            spine_x_vals.append(x_val)
                            spine_y_vals.append(y_val)
                    
                    if spine_x_vals and spine_y_vals:
                        spine_x = sum(spine_x_vals) / len(spine_x_vals)
                        spine_y = sum(spine_y_vals) / len(spine_y_vals)
                        spine_positions.append((spine_x, spine_y))
                    else:
                        spine_available = False
                        break
                else:
                    spine_available = False
                    break
        except Exception as e:
            print(f"Error extracting spine data: {e}")
            spine_available = False
        
        # Calculate head bearing
        if ear_distance >= minimum_ear_distance:
            # Use ear-based calculation
            # Vector from left ear to right ear
            ear_vector_x = average_right_ear[0] - average_left_ear[0]
            ear_vector_y = average_right_ear[1] - average_left_ear[1]
            
            # The head direction is perpendicular to the ear vector
            # For a mouse facing forward, if we go from left ear to right ear,
            # the forward direction is 90 degrees CLOCKWISE from that vector
            # In screen coordinates (y flipped), this becomes:
            head_vector_x = ear_vector_y   # Note: changed from -ear_vector_y
            head_vector_y = -ear_vector_x  # Note: changed from ear_vector_x
            
            # Calculate angle using atan2 with -y for video coordinates
            theta_rad = math.atan2(-head_vector_y, head_vector_x)
            theta_deg = math.degrees(theta_rad)
            theta_deg = theta_deg % 360
            
            # Validate with spine data if available
            if spine_available and len(spine_positions) >= 2:
                # Calculate spine direction vector
                spine_vector_x = spine_positions[0][0] - spine_positions[-1][0]  # spine_1 to spine_4
                spine_vector_y = spine_positions[0][1] - spine_positions[-1][1]
                
                # Calculate angle of spine direction
                spine_angle_rad = math.atan2(-spine_vector_y, spine_vector_x)
                spine_angle_deg = math.degrees(spine_angle_rad) % 360
                
                # Check angle difference
                angle_diff = abs(theta_deg - spine_angle_deg)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Only flip if angle difference is large AND majority of spine points suggest flip
                if angle_diff > 90:
                    # Check if majority of spine points are on the "wrong" side
                    # Calculate which side of the ear line each spine point is on
                    flip_votes = 0
                    for spine_pos in spine_positions:
                        # Vector from left ear to spine point
                        to_spine_x = spine_pos[0] - average_left_ear[0]
                        to_spine_y = spine_pos[1] - average_left_ear[1]
                        
                        # Cross product to determine which side of ear line
                        cross_product = ear_vector_x * to_spine_y - ear_vector_y * to_spine_x
                        
                        # If spine point is on the "back" side (negative cross product), vote for flip
                        if cross_product < 0:
                            flip_votes += 1
                    
                    # Only flip if majority of spine points vote for it
                    if flip_votes > len(spine_positions) / 2:
                        theta_deg = (theta_deg + 180) % 360
                        angle_correction_method = "ear_flip_corrected"
                    
        elif ear_distance < minimum_ear_distance and spine_available and len(spine_positions) >= 2:
            # Ears too close - use spine-based calculation
            angle_correction_method = "spine_based"
            
            # Find best fit line through all spine points
            spine_x = [pos[0] for pos in spine_positions]
            spine_y = [pos[1] for pos in spine_positions]
            
            # Validate spine data before polyfit
            if len(set(spine_x)) < 2:  # All x values are the same
                # Handle vertical line case
                if abs(spine_x[0] - spine_x[-1]) < 0.1:  # Nearly vertical
                    if spine_positions[0][1] < spine_positions[-1][1]:  # spine_1.y < spine_4.y (head is up)
                        theta_rad = math.pi / 2  # 90 degrees
                    else:
                        theta_rad = -math.pi / 2  # 270 degrees
                else:
                    # Fallback calculation
                    spine_vector_x = spine_positions[0][0] - spine_positions[-1][0]
                    spine_vector_y = spine_positions[0][1] - spine_positions[-1][1]
                    theta_rad = math.atan2(-spine_vector_y, spine_vector_x)
            else:
                # Use numpy polyfit to find line of best fit
                try:
                    # Filter out any remaining NaN values
                    valid_points = [(x, y) for x, y in zip(spine_x, spine_y) 
                                if np.isfinite(x) and np.isfinite(y)]
                    
                    if len(valid_points) >= 2:
                        spine_x_clean = [p[0] for p in valid_points]
                        spine_y_clean = [p[1] for p in valid_points]
                        
                        # Check for extreme values that might cause numerical issues
                        x_range = max(spine_x_clean) - min(spine_x_clean)
                        y_range = max(spine_y_clean) - min(spine_y_clean)
                        
                        if x_range > 1e-6 and not any(abs(x) > 1e6 for x in spine_x_clean):
                            coefficients = np.polyfit(spine_x_clean, spine_y_clean, 1)
                            slope = coefficients[0]
                            
                            # Determine direction based on spine_1 to spine_4
                            if spine_positions[0][0] > spine_positions[-1][0]:  # spine_1.x > spine_4.x
                                # Head is to the right of tail
                                theta_rad = math.atan(-slope)
                            else:
                                # Head is to the left of tail - rotate by 180°
                                theta_rad = math.atan(-slope) + math.pi
                        else:
                            # Fallback for extreme or degenerate cases
                            spine_vector_x = spine_positions[0][0] - spine_positions[-1][0]
                            spine_vector_y = spine_positions[0][1] - spine_positions[-1][1]
                            theta_rad = math.atan2(-spine_vector_y, spine_vector_x)
                    else:
                        # Not enough valid points
                        spine_vector_x = spine_positions[0][0] - spine_positions[-1][0]
                        spine_vector_y = spine_positions[0][1] - spine_positions[-1][1]
                        theta_rad = math.atan2(-spine_vector_y, spine_vector_x)
                        
                except Exception as e:
                    print(f"Polyfit error: {e}")
                    # Fallback calculation
                    spine_vector_x = spine_positions[0][0] - spine_positions[-1][0]
                    spine_vector_y = spine_positions[0][1] - spine_positions[-1][1]
                    theta_rad = math.atan2(-spine_vector_y, spine_vector_x)
            
            theta_deg = math.degrees(theta_rad)
            theta_deg = theta_deg % 360
            
        else:
            # Fallback to ear-based calculation without validation
            if ear_distance < minimum_ear_distance:
                angle_correction_method = "ears_too_close_no_spine"
            
            # Same calculation as corrected ear-based above
            ear_vector_x = average_right_ear[0] - average_left_ear[0]
            ear_vector_y = average_right_ear[1] - average_left_ear[1]
            
            head_vector_x = ear_vector_y
            head_vector_y = -ear_vector_x
            
            theta_rad = math.atan2(-head_vector_y, head_vector_x)
            theta_deg = math.degrees(theta_rad)
            theta_deg = theta_deg % 360
        
        # Calculate midpoint
        if angle_correction_method == "spine_based" and spine_available and len(spine_positions) >= 1:
            # Use spine_1 position as reference point when using spine-based angle
            midpoint_x = spine_positions[0][0]
            midpoint_y = spine_positions[0][1]
        else:
            # Use ear midpoint
            midpoint_x = (average_left_ear[0] + average_right_ear[0]) / 2
            midpoint_y = (average_left_ear[1] + average_right_ear[1]) / 2
        
        # Apply offset to midpoint in the direction of heading
        theta_rad = math.radians(theta_deg)
        eyes_offset = 40
        offset_x = eyes_offset * math.cos(theta_rad)
        offset_y = eyes_offset * math.sin(theta_rad)
        
        # Note: y offset is subtracted because sin is already accounting for flipped coords
        new_midpoint_x = midpoint_x + offset_x
        new_midpoint_y = midpoint_y - offset_y
        
        midpoint = (new_midpoint_x, new_midpoint_y)
        
        # Get cue presentation angle from mouse heading
        if self.rig_id == 1:
            self.port_angles = [64, 124, 184, 244, 304, 364] 
        elif self.rig_id == 2:
            self.port_angles = [240, 300, 360, 420, 480, 540]
        elif self.rig_id == 3:
            self.port_angles = [60, 120, 180, 240, 300, 360]
        else:
            raise ValueError(f"Invalid rig ID: {self.rig_id}")

        frame_height = 1080
        frame_width = 1280
        center_x = frame_width / 2
        center_y = frame_height / 2
        distance = (frame_height / 2) * 0.9

        self.port_coordinates = []
        for angle_deg in self.port_angles:
            angle_rad = np.deg2rad(angle_deg)
            
            x = int(center_x + distance * np.cos(angle_rad))
            y = int(center_y - distance * np.sin(angle_rad))
            
            self.port_coordinates.append((x, y))

        self.relative_angles = []
        mouse_heading_rad = np.deg2rad(theta_deg)

        for port_x, port_y in self.port_coordinates:
            vector_x = port_x - midpoint[0]
            vector_y = port_y - midpoint[1]

            port_angle_rad = math.atan2(-vector_y, vector_x)

            relative_angle_rad = port_angle_rad - mouse_heading_rad

            relative_angle_deg = math.degrees(relative_angle_rad) % 360

            self.relative_angles.append(relative_angle_deg)

        correct_port = trial["correct_port"]
        if correct_port == "audio-1":
            correct_port = 1
        port = int(correct_port) - 1
        cue_presentation_angle = self.relative_angles[port] % 360

        port_touched_angle = None
        if trial['next_sensor'] != {}:
            port_touched = trial['next_sensor'].get('sensor_touched')
            
            if port_touched != None:
                port_touched_angle = self.relative_angles[int(port_touched) - 1] % 360
                
        if cue_presentation_angle > 180:
            cue_presentation_angle -= 360
        elif cue_presentation_angle <= -180:
            cue_presentation_angle += 360

        if port_touched_angle != None:
            if port_touched_angle > 180:
                port_touched_angle -= 360
            elif port_touched_angle <= -180:
                port_touched_angle += 360

        # Return data
        angle_data = {
            "bearing": theta_deg,
            "midpoint": midpoint,
            "left_ear_likelihood": avg_left_ear_likelihood,
            "right_ear_likelihood": avg_right_ear_likelihood,
            "cue_presentation_angle": cue_presentation_angle,
            "port_touched_angle": port_touched_angle,
            "angle_correction_method": angle_correction_method,
            "ear_distance": ear_distance,
            "spine_data_available": spine_available
        }

        return angle_data
        
    def draw_LEDs(self, output_path, start=0, end=None):
        if self.rig_id == 1:
            port_angles = [64, 124, 184, 244, 304, 364]  # calibrated 14/2/24 with function at end of session class
        elif self.rig_id == 2:
            port_angles = [240, 300, 360, 420, 480, 540]
        elif self.rig_id == 3:
            self.port_angles = [60, 120, 180, 240, 300, 360]
        else:
            raise ValueError(f"Invalid rig ID: {self.rig_id}")

        # Prepare frame text for each trial
        frame_text = {}
        for i, trial in enumerate(self.trials):
            for frame in trial["video_frames"]:
                frame_text[frame] = {
                    "text": f"trial {i + 1}, cue: {trial['correct_port']}",
                    "cue": trial['correct_port']
                }

        output_folder = Path(output_path) / "drawn_videos"
        filename = f"{self.session_ID}_labelled_LEDs.mp4"
        output_filename = output_folder / filename

        # Ensure the output directory exists
        output_folder.mkdir(parents=True, exist_ok=True)

        cap = cv.VideoCapture(str(self.session_video))
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.session_video}")
            return

        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f"FPS: {fps}, Width: {frame_width}, Height: {frame_height}, frame count: {total_frame_count}")
        out = cv.VideoWriter(str(output_filename), fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("Error: VideoWriter failed to open.")
            return
        
        centre = (int(frame_width / 2), int(frame_height / 2))
        cue_position = (frame_height / 2) - 25

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if start > frame_count:
            raise Exception(f"start frame {start} is greater than frame count {frame_count}")
        if end is None:
            end = frame_count
        end = end if end < frame_count else frame_count

        # Progress tracking
        self.UP = "\033[1A"
        self.CLEAR = '\x1b[2K'

        frame_index = 0
        while frame_index < end:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from video.")
                break

            if frame_index >= start:
                # Update progress every 100 frames or so
                if frame_index % 100 == 0:
                    print(f"Processing frame {frame_index}/{end}", end=self.CLEAR + self.UP)

                # Check if current frame should have annotations
                if frame_index in frame_text:
                    frame = cv.putText(
                        frame, frame_text[frame_index]["text"], (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv.LINE_AA
                    )
                    port = frame_text[frame_index]["cue"]
                    try:
                        port = int(port) - 1
                        port_angle = port_angles[port]
                        x2 = int(centre[0] + cue_position * math.cos(math.radians(port_angle)))
                        y2 = int(centre[1] - cue_position * math.sin(math.radians(port_angle)))
                        # Draw a marker at the cue position
                        frame = cv.drawMarker(
                            frame, (x2, y2), (0, 255, 0), markerType=cv.MARKER_STAR,
                            markerSize=30, thickness=2, line_type=cv.LINE_AA
                        )
                    except ValueError:
                        frame = cv.putText(frame, "audio", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                # Write the processed frame to the output
                out.write(frame)

            frame_index += 1

        # Release resources
        cap.release()
        out.release()
        print(f"Processing complete. Video saved to {output_filename}")


    def load_data(self):
        # Load the NWB file:
        with NWBHDF5IO(str(self.nwb_file_path), 'r') as io:
            nwbfile = io.read()
            
            # Determine which processing module to look for
            if self.dlc_model_name:
                coords_module_name = f'behaviour_coords_{self.dlc_model_name}'
            else:
                coords_module_name = 'behaviour_coords'
            
            DLC_coords_present = coords_module_name in nwbfile.processing
            self.coords_module_name = coords_module_name  # Store for use in other methods
            
            self.last_timestamp = nwbfile.acquisition['scales'].timestamps[-1]
            session_metadata = nwbfile.experiment_description
            
            # session metadata looks like this: "phase:x;rig;y". Extract x and y:
            self.phase = str(session_metadata.split(";")[0].split(":")[1])
            self.rig_id = int(session_metadata.split(";")[1].split(":")[1])
            
            # Always load video path and timestamps - regardless of pickle loading
            if 'behaviour_video' in nwbfile.acquisition:
                self.session_video_object = nwbfile.acquisition['behaviour_video']
                self.session_video = self.session_directory / Path(self.session_video_object.external_file[0])
                self.video_timestamps = self.session_video_object.timestamps[:]
            else:
                print("WARNING: No 'behaviour_video' found in NWB acquisition")
                self.session_video = None
                self.session_video_object = None
                self.video_timestamps = None

        # check if dlc coords are in nwb file already:
        if not DLC_coords_present:
            # Need to find the correct CSV file based on model name
            if self.dlc_model_name:
                # Find CSV file with specific model name
                csv_files = list(self.session_directory.glob("*.csv"))
                self.DLC_coords_path = None
                for csv_file in csv_files:
                    if self.dlc_model_name in csv_file.name and 'DLC' in csv_file.name:
                        self.DLC_coords_path = csv_file
                        break
                if not self.DLC_coords_path:
                    raise FileNotFoundError(f"No DLC CSV file found for model {self.dlc_model_name}")
            else:
                # Use default from session_dict
                self.DLC_coords_path = Path(self.session_dict.get('processed_data', {}).get('DLC', {}).get('coords_csv'))
            
            self.DLC_coords = pd.read_csv(self.DLC_coords_path, header=[1, 2], index_col=0)
            # if the behaviour coordinates data hasn't been added yet, add the data to the nwb file:
            self.add_DLC_coords_to_nwb()

    def add_DLC_coords_to_nwb(self):
        """
        Add DeepLabCut coordinate data to the NWB file, ensuring that the DLC data
        length exactly matches the video timestamps length to prevent dimension warnings.
        """
        # Load the DLC coords:
        self.DLC_coords = pd.read_csv(self.DLC_coords_path, header=[1, 2], index_col=0)

        with NWBHDF5IO(str(self.nwb_file_path), 'a') as io:
            nwbfile = io.read()

            video_data_name = 'behaviour_video'
            if video_data_name in nwbfile.acquisition:
                video_data = nwbfile.acquisition[video_data_name]
                video_timestamps = video_data.timestamps[:]
                
                # Get the number of video timestamps - this is our target length
                num_timestamps = len(video_timestamps)
                print(f"Number of video timestamps: {num_timestamps}")
                print(f"Number of DLC frames (original): {len(self.DLC_coords)}")

                # Create a new processing module for the DLC coords with model-specific name:
                behaviour_module = ProcessingModule(name=self.coords_module_name, 
                                                description=f'DLC coordinates for behaviour{" - " + self.dlc_model_name if self.dlc_model_name else ""}')
                nwbfile.add_processing_module(behaviour_module)

                for body_part in self.DLC_coords.columns.levels[0]:
                    # Extract the data for this body part and truncate to exactly match timestamps
                    body_part_data = self.DLC_coords[body_part].iloc[:num_timestamps]
                    
                    # Ensure we have exactly the right number of rows
                    if len(body_part_data) > num_timestamps:
                        body_part_data = body_part_data.iloc[:num_timestamps]
                        print(f"Truncated {body_part} from {len(self.DLC_coords)} to {num_timestamps} frames")
                    elif len(body_part_data) < num_timestamps:
                        print(f"WARNING: {body_part} has only {len(body_part_data)} frames, less than {num_timestamps} timestamps")
                        # Pad with NaN if needed (though this shouldn't happen with your data)
                        # This ensures exact length match
                        padding_needed = num_timestamps - len(body_part_data)
                        padding = pd.DataFrame(
                            np.full((padding_needed, body_part_data.shape[1]), np.nan),
                            columns=body_part_data.columns,
                            index=range(len(body_part_data), num_timestamps)
                        )
                        body_part_data = pd.concat([body_part_data, padding])
                    
                    # Convert to numpy array - should be exactly (num_timestamps, 3)
                    data = body_part_data.values
                    
                    # print(f"Body part: {body_part}")
                    # print(f"  Final data shape: {data.shape}")
                    # print(f"  Timestamps shape: {video_timestamps.shape}")
                    # print(f"  Dimension match: {data.shape[0] == len(video_timestamps)}")
                    
                    # Verify exact dimension match
                    assert data.shape[0] == len(video_timestamps), f"Dimension mismatch for {body_part}: {data.shape[0]} != {len(video_timestamps)}"
                    
                    # Create the TimeSeries with exactly matching dimensions
                    ts = TimeSeries(
                        name=body_part,
                        data=data,  # Should be exactly (num_timestamps, 3)
                        unit='pixels',
                        timestamps=video_timestamps,  # Should be (num_timestamps,)
                        description=f"Coordinates and likelihood for {body_part} [x, y, likelihood]"
                    )

                    behaviour_module.add_data_interface(ts)

                io.write(nwbfile)
                print("DLC coordinates successfully added to NWB file")
        

    def frametime_to_index(self, frametimes):
        # Initialize the dictionary for fast lookup
        index_frametimes = {}


        for i, (frame_ID, time) in enumerate(frametimes.items()):
            if self.is_number(frame_ID):
                # Use bisect to find the insertion index
                index = bisect.bisect_left(self.timestamps, float(time))

                # Check that the index is within the range of self.timestamps
                if index < len(self.timestamps):
                    # If the timestamp at the derived index is greater than the desired time,
                    # and we're not looking at the first element,
                    # move the index back to align with the "less than or equal" requirement
                    if index > 0 and self.timestamps[index] > time:
                        index -= 1

                # Create a small dictionary for the frame information and assign it to the index key
                frame_info = {'frame_ID': frame_ID, 'time': time, "frame_no": i}
                index_frametimes[index] = frame_info

        # print(f"Max index in frametimes: {max(index_frametimes.keys())}")
        # print(f"Max index in timestamps: {len(self.timestamps)}")

        return index_frametimes

    def add_video_data_to_trials(self):
            # get each trial object and attach a list of the relevant video frames between the start and end of the trial,
            # and also a dataframe of the DLC data for the same time period, using those frame times as the indeices for the slice.
            
            # Video path should already be loaded in load_data(), but check just in case
            if not hasattr(self, 'session_video') or self.session_video is None:
                print("WARNING: Video path not loaded, attempting to load from NWB file")
                with NWBHDF5IO(self.nwb_file_path, 'r') as io:
                    nwbfile = io.read()
                    if 'behaviour_video' in nwbfile.acquisition:
                        self.session_video_object = nwbfile.acquisition['behaviour_video']
                        self.session_video = self.session_directory / Path(self.session_video_object.external_file[0])
                        self.video_timestamps = self.session_video_object.timestamps[:]
                    else:
                        print("ERROR: Cannot find behaviour_video in NWB file")
                        return
            
            with NWBHDF5IO(self.nwb_file_path, 'r') as io:
                nwbfile = io.read()
                
                for j, trial in enumerate(self.trials):
                    if trial["phase"] != "1" or trial["phase"] != "2":
                        start = trial["cue_start"]
                        if trial["next_sensor"].get("sensor_start") != None:
                            end = trial["next_sensor"]["sensor_start"]
                        else:
                            end = self.trials[j+1]["cue_start"] if j+1 < len(self.trials) else self.last_timestamp

                    if trial["phase"] == "1" or trial["phase"] == "2":
                        start = trial["cue_start"]
                        end = self.trials[j+1]["cue_start"] if j+1 < len(self.trials) else self.last_timestamp

                    video_frames = []
                    
                    # use bisect to find the frametimes from the indexed_frametimes dictionary
                    start_index = np.searchsorted(self.video_timestamps, start, side='left')
                    if end != None:
                        end_index = np.searchsorted(self.video_timestamps, end, side='left')
                    else:
                        end_index = len(self.video_timestamps) - 1

                    # for each index in between the start and end, see if it's in the indexed_frametimes dictionary, and if it is, add it to the video_frames list.
                    # bug happened here where I was appending the frame_ID not the frame number.
                    # when the video processor later went to grab the appropriate frame, it grabbed the wrong frame, since it needed the index in the video.
                    for i in range(start_index, end_index):
                        video_frames.append(i)
                    
                    if len(video_frames) != 0:
                        if self.coords_module_name in nwbfile.processing:  # Use stored module name
                            behaviour_module = nwbfile.processing[self.coords_module_name]
                            
                            # Initialize an empty dictionary to store DLC data for the trial
                            trial_DLC_data = {}

                            for time_series_name in behaviour_module.data_interfaces:
                                ts = behaviour_module.data_interfaces[time_series_name]

                                # Get the timestamps and data for the body part
                                timestamps = ts.timestamps[:]
                                data = ts.data[:]

                                # Find indices for the trial time range
                                start_index = np.searchsorted(timestamps, start, side='left')
                                end_index = np.searchsorted(timestamps, end, side='left')

                                # Slice the data and timestamps for the trial
                                trial_data = data[start_index:end_index]
                                trial_timestamps = timestamps[start_index:end_index]

                                trial_DLC_data[time_series_name] = trial_data
                            
                            # Initialize a dictionary to hold the data for DataFrame construction
                            dlc_data_dict = {}

                            for body_part, data in trial_DLC_data.items():
                                # Assuming 'data' is a 2D array with columns for x, y, and confidence
                                x, y, likelihood = data.transpose()  # Transpose to separate the columns

                                # Populate the dictionary for DataFrame creation
                                dlc_data_dict[(body_part, 'x')] = x
                                dlc_data_dict[(body_part, 'y')] = y
                                dlc_data_dict[(body_part, 'likelihood')] = likelihood

                            if len(video_frames) > data.shape[0]:
                                video_frames = video_frames[:data.shape[0]]

                            # Create a MultiIndex DataFrame from the dictionary
                            dlc_df = pd.DataFrame(dlc_data_dict, index=video_frames)

                            dlc_df['timestamps'] = trial_timestamps
                            trial["video_frames"] = video_frames
                            trial["DLC_data"] = dlc_df
                    else:
                        trial["video_frames"] = video_frames
                        trial["DLC_data"] = None

                # also add the DLC data for the same time period, if it's been found:
                # if self.DLC_coords is not None:
                #     if len(video_frames) != 0:
                #         try:
                #             trial["DLC_data"] = self.DLC_coords.loc[video_frames[0]:video_frames[-1]]
                #         except IndexError:
                #             print(f"max dlc_coord = {len(self.DLC_coords)}")
                #             print(f"max video_frames = {len(self.indexed_frametimes)}")
                #             print(list(self.indexed_frametimes.keys())[-1])
                #             print(f"num trials = {len(self.trials)}")
                #             print(start, end)
                #             raise Exception(f"error happening on trial {j}")
                #     else:
                #         trial["DLC_data"] = None

    def find_file(self, directory, tag):
        file_found = False
        for file in directory.glob('*'):
            if not file_found:
                if tag in file.name:
                    file_found = True
                    return file
        if not file_found:
            raise Exception(f"'{tag}' not found in {directory}")

    def find_dir(self, directory):
        """
        if there is only one directory in the directory, return that directory
        """
        dir_found = False
        for file in directory.glob('*'):
            if not dir_found:
                # if dir and does not contain "DLC":    Makes sure it's not detecting the DLC folder that's generated.
                if file.is_dir() and "DLC" not in file.name:
                    dir_found = True
                    return file
        if not dir_found:
            raise Exception(f"No OEAB directory found in {directory}")

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
    def calibrate_port_angles(self, calibration_offset=0, output_file_path="test_output"):
        """
        Calculate and visualise the angles of 6 ports from the platform centre.
        
        Parameters:
        -----------
        calibration_offset : float
            Rotation offset in degrees to align the drawn lines with actual ports
        output_file_path : str
            Directory path where the output image will be saved
            
        Returns:
        --------
        list
            List of 6 calibrated angles (in degrees) for each port
        """
        # Initialise port angles (0° is rightward, angles increase anticlockwise)
        port_angles = [0, 60, 120, 180, 240, 300]
        
        # Extract first frame from trial 1
        frame_number = self.trials[1]["video_frames"][0]
        
        # Open video capture
        video_capture = cv.VideoCapture(str(self.session_video))
        if not video_capture.isOpened():
            print(f"Error: Unable to open video file {self.session_video}")
            return None
        
        # Get video dimensions
        frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        centre_x = frame_width // 2
        centre_y = frame_height // 2
        
        # Navigate to specified frame
        video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video_capture.read()
        
        if not success:
            print(f"Error: Failed to read frame at position {frame_number}")
            video_capture.release()
            return None
        
        # Draw lines from centre to each port position
        line_length = 400  # Length of indicator lines in pixels
        for port_index, angle in enumerate(port_angles):
            # Calculate line endpoint coordinates
            adjusted_angle = angle + calibration_offset
            end_x = int(centre_x + line_length * math.cos(math.radians(adjusted_angle)))
            end_y = int(centre_y - line_length * math.sin(math.radians(adjusted_angle)))
            
            # Draw line from centre to port direction
            cv.line(frame, (centre_x, centre_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Label each line with port number (1-indexed)
            port_label = str(port_index + 1)
            cv.putText(frame, port_label, (end_x, end_y), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        
        # Save annotated frame to file
        output_directory = Path(output_file_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_filepath = output_directory / 'angles.jpg'
        cv.imwrite(str(output_filepath), frame)
        
        # Display image in notebook using matplotlib
        import matplotlib.pyplot as plt
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Create figure and display
        plt.figure(figsize=(10, 8))
        plt.imshow(frame_rgb)
        plt.title(f'Port Calibration (Offset: {calibration_offset}°)')
        plt.axis('off')
        plt.show()
        
        # Clean up resources
        video_capture.release()
        
        # Calculate and return calibrated angles
        calibrated_angles = [(angle + calibration_offset) % 360 for angle in port_angles]
        print(f"Calibrated angles: {calibrated_angles}")
        
        return calibrated_angles

    def print_trial_summary(self, output_file=None, include_angle_details=True):
        """
        Generate a comprehensive human-readable summary of all trials in the session.
        
        Parameters:
        -----------
        output_file : str or Path, optional
            If provided, writes the summary to this file instead of printing to console
        include_angle_details : bool, default=True
            Whether to include detailed angle analysis information
            
        Returns:
        --------
        str
            The formatted summary string
        """
        summary_lines = []
        
        # Session header
        summary_lines.append("=" * 80)
        summary_lines.append(f"SESSION SUMMARY: {self.session_ID}")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Session Directory: {self.session_directory}")
        summary_lines.append(f"Phase: {self.phase}")
        summary_lines.append(f"Rig ID: {self.rig_id}")
        summary_lines.append(f"Last Timestamp: {self.last_timestamp:.2f}s")
        summary_lines.append(f"Total Trials: {len(self.trials)}")
        summary_lines.append(f"Analysis Pickle: {self.analysis_pickle_path}")
        
        if hasattr(self, 'session_video') and self.session_video:
            summary_lines.append(f"Video File: {self.session_video.name}")
            if hasattr(self, 'video_timestamps') and self.video_timestamps is not None:
                summary_lines.append(f"Video Duration: {len(self.video_timestamps)} frames")
        
        summary_lines.append("")
        
        # Trial-by-trial breakdown
        for i, trial in enumerate(self.trials):
            summary_lines.append("-" * 60)
            summary_lines.append(f"TRIAL {i + 1}")
            summary_lines.append("-" * 60)
            
            # Basic trial information
            summary_lines.append(f"Trial Number: {trial.get('trial_no', 'N/A')}")
            summary_lines.append(f"Phase: {trial.get('phase', 'N/A')}")
            summary_lines.append(f"Correct Port: {trial.get('correct_port', 'N/A')}")
            summary_lines.append(f"Outcome: {trial.get('outcome', 'N/A')}")
            
            # Timing information
            if 'cue_start' in trial:
                summary_lines.append(f"Cue Start: {trial['cue_start']:.3f}s")
            if 'cue_end' in trial:
                summary_lines.append(f"Cue End: {trial['cue_end']:.3f}s")
            if 'start_time' in trial:
                summary_lines.append(f"Start Time: {trial['start_time']:.3f}s")
            if 'end_time' in trial:
                summary_lines.append(f"End Time: {trial['end_time']:.3f}s")
                duration = trial['end_time'] - trial.get('start_time', trial.get('cue_start', 0))
                summary_lines.append(f"Duration: {duration:.3f}s")
            
            # Sensor information
            if 'next_sensor' in trial and trial['next_sensor']:
                sensor_info = trial['next_sensor']
                summary_lines.append(f"Sensor Response:")
                for key, value in sensor_info.items():
                    if key == 'sensor_touched':
                        summary_lines.append(f"  Port Touched: {value}")
                    elif key == 'sensor_start':
                        summary_lines.append(f"  Response Time: {value:.3f}s")
                    elif key == 'sensor_end':
                        summary_lines.append(f"  Response End: {value:.3f}s")
                    else:
                        summary_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                summary_lines.append("Sensor Response: No sensor interaction")
            
            # Reward information
            if 'reward_touch' in trial and trial['reward_touch']:
                reward_info = trial['reward_touch']
                summary_lines.append(f"Reward Touch: Present")
                if 'start' in reward_info:
                    summary_lines.append(f"  Reward Start: {reward_info['start']:.3f}s")
                if 'end' in reward_info:
                    summary_lines.append(f"  Reward End: {reward_info['end']:.3f}s")
            else:
                summary_lines.append("Reward Touch: None")
            
            # Video frame information (condensed)
            if 'video_frames' in trial and trial['video_frames']:
                frames = trial['video_frames']
                summary_lines.append(f"Video Frames: {len(frames)} frames (#{frames[0]} to #{frames[-1]})")
            else:
                summary_lines.append("Video Frames: No video data")
            
            # DLC data information (condensed)
            if 'DLC_data' in trial and trial['DLC_data'] is not None:
                dlc_data = trial['DLC_data']
                summary_lines.append(f"DLC Data: {len(dlc_data)} rows × {len(dlc_data.columns)} columns")
                
                # Get body parts tracked
                if hasattr(dlc_data.columns, 'levels') and len(dlc_data.columns.levels) > 0:
                    body_parts = dlc_data.columns.levels[0].tolist()
                    summary_lines.append(f"  Body Parts: {', '.join(body_parts)}")
                
                # Show data quality summary
                if 'timestamps' in dlc_data.columns:
                    time_span = dlc_data['timestamps'].max() - dlc_data['timestamps'].min()
                    summary_lines.append(f"  Time Span: {time_span:.3f}s")
                
                # Show likelihood statistics for key body parts
                key_parts = ['left_ear', 'right_ear', 'nose'] if hasattr(dlc_data.columns, 'levels') else []
                for part in key_parts:
                    if part in dlc_data.columns.levels[0]:
                        likelihood_col = (part, 'likelihood')
                        if likelihood_col in dlc_data.columns:
                            mean_likelihood = dlc_data[likelihood_col].mean()
                            summary_lines.append(f"  {part.replace('_', ' ').title()} Likelihood: {mean_likelihood:.2f}")
            else:
                summary_lines.append("DLC Data: No tracking data")
            
            # Angle analysis information (if available and requested)
            if include_angle_details and 'turn_data' in trial and trial['turn_data']:
                angle_data = trial['turn_data']
                summary_lines.append(f"Angle Analysis:")
                summary_lines.append(f"  Mouse Bearing: {angle_data.get('bearing', 'N/A'):.1f}°")
                summary_lines.append(f"  Cue Presentation Angle: {angle_data.get('cue_presentation_angle', 'N/A'):.1f}°")
                
                if angle_data.get('port_touched_angle') is not None:
                    summary_lines.append(f"  Port Touched Angle: {angle_data['port_touched_angle']:.1f}°")
                
                summary_lines.append(f"  Correction Method: {angle_data.get('angle_correction_method', 'N/A')}")
                summary_lines.append(f"  Ear Distance: {angle_data.get('ear_distance', 'N/A'):.1f} pixels")
                summary_lines.append(f"  Left Ear Likelihood: {angle_data.get('left_ear_likelihood', 'N/A'):.2f}")
                summary_lines.append(f"  Right Ear Likelihood: {angle_data.get('right_ear_likelihood', 'N/A'):.2f}")
                summary_lines.append(f"  Spine Data Available: {angle_data.get('spine_data_available', 'N/A')}")
            elif include_angle_details:
                summary_lines.append("Angle Analysis: No angle data")
            
            # Additional trial-specific fields
            other_fields = []
            skip_fields = {'trial_no', 'phase', 'correct_port', 'outcome', 'cue_start', 'cue_end', 
                        'start_time', 'end_time', 'next_sensor', 'reward_touch', 'video_frames', 
                        'DLC_data', 'turn_data', 'start', 'next_sensor_time'}
            
            for key, value in trial.items():
                if key not in skip_fields:
                    if isinstance(value, (list, dict, np.ndarray)):
                        if isinstance(value, list):
                            other_fields.append(f"{key.replace('_', ' ').title()}: List with {len(value)} items")
                        elif isinstance(value, dict):
                            other_fields.append(f"{key.replace('_', ' ').title()}: Dict with {len(value)} keys")
                        else:
                            other_fields.append(f"{key.replace('_', ' ').title()}: Array {value.shape}")
                    else:
                        other_fields.append(f"{key.replace('_', ' ').title()}: {value}")
            
            if other_fields:
                summary_lines.append("Other Information:")
                for field in other_fields:
                    summary_lines.append(f"  {field}")
            
            summary_lines.append("")  # Empty line between trials
        
        # Summary statistics
        summary_lines.append("=" * 80)
        summary_lines.append("SUMMARY STATISTICS")
        summary_lines.append("=" * 80)
        
        # Count outcomes
        outcomes = [trial.get('outcome', 'Unknown') for trial in self.trials]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        summary_lines.append("Trial Outcomes:")
        for outcome, count in outcome_counts.items():
            percentage = (count / len(self.trials)) * 100
            summary_lines.append(f"  {outcome}: {count} ({percentage:.1f}%)")
        
        # Count correct ports
        correct_ports = [trial.get('correct_port', 'Unknown') for trial in self.trials]
        port_counts = {}
        for port in correct_ports:
            port_counts[port] = port_counts.get(port, 0) + 1
        
        summary_lines.append("Correct Port Distribution:")
        for port, count in sorted(port_counts.items()):
            percentage = (count / len(self.trials)) * 100
            summary_lines.append(f"  Port {port}: {count} ({percentage:.1f}%)")
        
        # Timing statistics
        durations = []
        for trial in self.trials:
            if 'start_time' in trial and 'end_time' in trial:
                duration = trial['end_time'] - trial['start_time']
                durations.append(duration)
            elif 'cue_start' in trial and 'end_time' in trial:
                duration = trial['end_time'] - trial['cue_start']
                durations.append(duration)
        
        if durations:
            summary_lines.append("Trial Duration Statistics:")
            summary_lines.append(f"  Mean: {np.mean(durations):.2f}s")
            summary_lines.append(f"  Median: {np.median(durations):.2f}s")
            summary_lines.append(f"  Min: {np.min(durations):.2f}s")
            summary_lines.append(f"  Max: {np.max(durations):.2f}s")
            summary_lines.append(f"  Standard Deviation: {np.std(durations):.2f}s")
        
        # Angle analysis summary (if available)
        if include_angle_details:
            angle_methods = []
            cue_angles = []
            for trial in self.trials:
                if 'turn_data' in trial and trial['turn_data']:
                    angle_data = trial['turn_data']
                    if 'angle_correction_method' in angle_data:
                        angle_methods.append(angle_data['angle_correction_method'])
                    if 'cue_presentation_angle' in angle_data:
                        cue_angles.append(angle_data['cue_presentation_angle'])
            
            if angle_methods:
                method_counts = {}
                for method in angle_methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                summary_lines.append("Angle Correction Methods:")
                for method, count in method_counts.items():
                    percentage = (count / len(angle_methods)) * 100
                    summary_lines.append(f"  {method}: {count} ({percentage:.1f}%)")
            
            if cue_angles:
                summary_lines.append("Cue Presentation Angles:")
                summary_lines.append(f"  Mean: {np.mean(cue_angles):.1f}°")
                summary_lines.append(f"  Median: {np.median(cue_angles):.1f}°")
                summary_lines.append(f"  Standard Deviation: {np.std(cue_angles):.1f}°")
        
        summary_lines.append("=" * 80)
        
        # Join all lines
        summary_text = "\n".join(summary_lines)
        
        # Output to file or console
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(summary_text)
            print(f"Trial summary saved to: {output_path}")
        else:
            print(summary_text)
        
        return summary_text

def main():

    cohort = Cohort_folder(r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment", multi=True, OEAB_legacy=False, use_existing_cohort_info=True)

    test_dir = cohort.get_session("250516_130123_mtao101-3c")
    # test_dir = cohort.get_session("240323_164315")

    session = Session(test_dir)

    session.print_trial_summary('/lmb/home/srogers/Dev/projects/hex_behav_analysis/temp_output/session_summary.txt')



if __name__ == "__main__":
    main()


