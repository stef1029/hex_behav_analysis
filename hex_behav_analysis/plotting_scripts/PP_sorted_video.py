import cv2 as cv
import json
from utils.Cohort_folder import Cohort_folder
from utils.Session import Session
from pathlib import Path
import numpy as np
import multiprocessing as mp
import time
import math
import subprocess as sp
import os

cohort_directory = Path(r"/cephfs2/srogers/240207_Dans_data")

cohort = Cohort_folder(cohort_directory)

info = cohort.cohort
phases = cohort.phases()

session_directories = []
videos = []
for session in phases["9"]:
    mouse = phases["9"][session]["mouse"]
    if mouse == "WTJP239-4b":
        session_directories.append(Path(phases["9"][session]["path"]))
        
        # append appropriate video path to videos list from info dict:
        videos.append(info["mice"][mouse]["sessions"][session]["raw_data"]["raw_video"])

videos = ["/cephfs2/srogers/test_output/LEDs.mp4"]

# session_directories = [session_directories[0]]

bins = {i: [] for i in range(-180, 180, 30)}

for i, directory in enumerate(session_directories):
    session = Session(directory)
    trials = session.trials

    for trial in trials:
        if trial["turn_data"] != None:
            angle = trial["turn_data"]["cue_presentation_angle"]
            for bin in bins:
                if angle < bin + 30 and angle >= bin:
                    bins[bin].append((i, trial))    # add a session cap index to identify where the trial comes from.

for bin in bins:
    # sort trials by presentation angle:
    bins[bin].sort(key=lambda x: x[1]["turn_data"]["cue_presentation_angle"])
    
# now you have bins of all the trials with turns to that particular angle.

def process_bin(bin):
    filename = f"/cephfs2/srogers/test_output/bin_{bin}.mp4"

    # Initialize video writer
    out = None  # Initialize out here to handle it based on the first successful VideoCapture
    
    for session_index, trial in bins[bin][:4]:

        trial_time = trial["end_time"] - trial["start_time"]
        success = True if int(trial["correct_port"]) == int(trial["next_sensor_ID"]) else False
        
        trial_count = 0

        if success and trial_time < 1.5 and trial_count < 4:
            trial_count += 1
            video_path = videos[session_index]
            cap = cv.VideoCapture(str(video_path))

            fps = cap.get(cv.CAP_PROP_FPS)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*'mp4v') 

            scale = 1.25
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            if not out:  # Setup video writer upon the first successful VideoCapture
                out = cv.VideoWriter(filename, fourcc, fps, (new_width, new_height))
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
            
            for frame_id in trial["video_frames"]:
                # Set the next frame to read from the video capture
                cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_id))
                
                # Read the frame
                ret, frame = cap.read()
                if ret:

                    # draw the ears on the frame
                    left_ear_coords = (int(trial["DLC_data"]["left_ear"]["x"].loc[int(frame_id)]), int(trial["DLC_data"]["left_ear"]["y"].loc[int(frame_id)]))
                    right_ear_coords = (int(trial["DLC_data"]["right_ear"]["x"].loc[int(frame_id)]), int(trial["DLC_data"]["right_ear"]["y"].loc[int(frame_id)]))
                    
                    try:
                        frame = cv.drawMarker(frame, left_ear_coords, (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv.LINE_AA)
                        frame = cv.drawMarker(frame, right_ear_coords, (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv.LINE_AA)
                        midpoint = trial['turn_data']['midpoint']
                        mouse_bearing = trial['turn_data']['bearing']

                        line_length = 50
                        x2 = int(midpoint[0] + line_length * math.cos(math.radians(mouse_bearing)))
                        y2 = int(midpoint[1] - line_length * math.sin(math.radians(mouse_bearing)))

                        frame = cv.line(frame, (int(midpoint[0]), int(midpoint[1])), (x2, y2), (0, 255, 0), 2)
                    except cv.error:
                        print("error")
                        pass

                    # enlarge frame:

                    large_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                    x_offset = (new_width - frame_width) // 2
                    y_offset = (new_height - frame_height) // 2

                    large_canvas[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
                    frame = large_canvas

                    center = (new_width // 2, new_height // 2)
                    # calculate the offset midpoint coords for the new large frame:
                    offset_midpoint = (midpoint[0] + x_offset, midpoint[1] + y_offset)

                    # shift midpoint to centre of frame:
                    tx = center[0] - offset_midpoint[0]
                    ty = center[1] - offset_midpoint[1]

                    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                    frame = cv.warpAffine(frame, translation_matrix, (new_width, new_height))

                    #rotate frame:
                    rotation_matrix = cv.getRotationMatrix2D(center, (-mouse_bearing) + 90, 1.0)

                    frame = cv.warpAffine(frame, rotation_matrix, (new_width, new_height))


                    # write turn angle on the frame
                    frame = cv.putText(frame, f"Turn angle: {trial['turn_data']['cue_presentation_angle']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    
                    # Write the frame to the output file
                    out.write(frame)
                else:
                    print("Error: Could not read frame.")
                    continue
            
            cap.release()  # Release the VideoCapture of this iteration
    
    if out:
        out.release()

def combine_output_files(output_file_name, bins, folder):
    # Create a list of output files and store the file names in a txt file
    folder = str(folder)
    num_processes = mp.cpu_count()
    list_of_output_files = [f"{folder}/bin_{bin}.mp4" for bin in bins]
    with open(f"{folder}/list_of_output_files.txt", "w") as f:
        for path in list_of_output_files:
            f.write(f"file {path} \n")

    # use ffmpeg to combine the video output files
    ffmpeg_cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i {folder}/list_of_output_files.txt -vcodec copy {folder}/{output_file_name}"
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    # for f in list_of_output_files:
    #     os.remove(f)
    os.remove(f"{folder}/list_of_output_files.txt")
    

if __name__ == "__main__":

    processes = 112
    print("Processing videos...")
    start = time.perf_counter()
    with mp.Pool(processes) as pool:
        pool.map(process_bin, bins)

    print("Combining output files...")
    combine_output_files("combined_output.mp4", bins, Path("/cephfs2/srogers/test_output"))

    print("Processing complete.")
    # print how long it took in minutes and seconds, rounded:
    print(f"Took {round((time.perf_counter() - start)/60)} minutes and {round((time.perf_counter() - start)%60)} seconds.")
    