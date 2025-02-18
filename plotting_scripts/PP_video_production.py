import cv2 as cv
from pathlib import Path
import multiprocessing as mp
# from Scripts.DAQ_plot import DAQ_plot
import numpy as np

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

class Video_maker:
    def __init__(self, session):
        # Create a session object to load the data
        self.session = session
        self.session_directory = self.session.session_directory

        # Load the trials
        self.trials = self.session.trials
        self.video_timestamps = self.session.video_timestamps
        self.rig_id = self.session.rig_id
        if self.rig_id == 1:
            self.port_angles = [64, 124, 184, 244, 304, 364] # calibrated 14/2/24 with function at end of session class
        elif self.rig_id == 2:
            self.port_angles = [240, 300, 360, 420, 480, 540]

        # Load the session video
        self.session_video = self.session.session_video

        self.make_frametext()

    def save_video(self, trial_no):
        # Path for the output video file
        filename = str(self.session_directory / f"trial_{trial_no}.mp4")
        
        # Retrieve the frame IDs for the specified trial
        frames = [int(frame) for frame in self.trials[trial_no]["video_frames"]]
        
        # Open the main session video
        cap = cv.VideoCapture(str(self.session_video))
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Get video properties to recreate it
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'mp4v') 
        out = cv.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        # Ensure frame IDs are sorted
        frames.sort()
        
        for frame_id in frames:
            # Set the next frame to read from the video capture
            cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_id))
            
            # Read the frame
            ret, frame = cap.read()
            if ret:
                # Write the frame to the output file
                out.write(frame)
            else:
                print(f"Skipping frame {frame_id}, unable to read.")

        # Release everything when job is finished
        cap.release()
        out.release()
        cv.destroyAllWindows()

    def make_frametext(self):
        video_length = len(self.video_timestamps)
        proportion_to_process = 0.1
        frames_to_process = int(video_length * proportion_to_process)
        video_frames = {}
        for i, trial in enumerate(self.trials[:frames_to_process]):
            cue_end_frame = np.searchsorted(self.video_timestamps, trial['cue_end'])
            for frame in trial["video_frames"]:
                video_frames[int(frame)] = {'text': f"Trial {i}, port: {trial['correct_port']}", 'cue_frames': 0, 'cue_angle': 0}
            for frame in trial['video_frames']:
                if frame < cue_end_frame:
                    video_frames[int(frame)]['cue_frames'] = 1
                    video_frames[int(frame)]['cue_angle'] = self.port_angles[int(trial['correct_port']) - 1]

        # sort by keys:
        video_frames = {k: video_frames[k] for k in sorted(video_frames)}

        cap = cv.VideoCapture(str(self.session_video))
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Get properties of the video
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        center_x, center_y = frame_width // 2, frame_height // 2

        self.UP = "\033[1A"; self.CLEAR = '\x1b[2K'
        
        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v') # Adjust codec as needed
        out = cv.VideoWriter('output_video_path.mp4', fourcc, fps, (frame_width,frame_height))

        current_frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {current_frame_index}/{frames_to_process}")
            print(self.UP, end = self.CLEAR)

            # If the current frame index is in video_frames, draw a dot
            if current_frame_index in video_frames:
                # Parameters: image, center_coordinates, radius, color, thickness
                # Adjust dot properties as needed
                cv.putText(frame, video_frames[current_frame_index]['text'], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                if video_frames[current_frame_index]['cue_frames']:
                    dot_x = int(center_x + (int(frame_height/2)*0.9) * np.cos(np.radians(video_frames[current_frame_index]['cue_angle'])))
                    dot_y = int(center_y - (int(frame_height/2)*0.9) * np.sin(np.radians(video_frames[current_frame_index]['cue_angle'])))
                    cv.circle(frame, (dot_x, dot_y), 20, (0, 255, 0), -1)
                    
            
            # Write the frame (modified or unmodified) to the output video
                out.write(frame)

            current_frame_index += 1
            if current_frame_index >= frames_to_process:
                break

        cap.release()
        out.release()


    # def test_plot(self):
    #     # create a DAQ_plot object and pass it the processed_DAQ_data and the session_directory
    #     DAQ_plot(self.processed_DAQ_data, self.session_directory, debug = True, trials = self.trials)



if __name__ == "__main__":
    # Example usage

    cohort_directory = Path("/cephfs2/srogers/March_training")

    cohort = Cohort_folder(cohort_directory, multi = True)
    cohort_info = cohort.cohort
    phases = cohort.phases()

    cue_group_500 = [session for session in phases['9c'] if session[:6] == '240325']
    
    session = Session(cohort.get_session(cue_group_500[0]))

    video_maker = Video_maker(session)
