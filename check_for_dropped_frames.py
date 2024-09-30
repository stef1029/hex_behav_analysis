import json

def check_dropped_frames(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the frame_IDs list
    frame_ids = data.get("frame_IDs", [])
    
    # Check if the list is not empty
    if not frame_ids:
        print("The frame_IDs list is empty.")
        return
    
    # Calculate the expected total number of frames
    expected_frames = len(frame_ids)
    
    # Calculate the actual number of frames (from the last value in the list)
    actual_frames = frame_ids[-1] + 1  # +1 because indices start at 0
    
    # Calculate the number of dropped frames
    dropped_frames = actual_frames - expected_frames
    
    # Calculate the percentage of dropped frames
    dropped_percentage = (dropped_frames / actual_frames) * 100 if actual_frames > 0 else 0
    
    # Print results
    if dropped_frames > 0:
        print(f"Dropped frames detected: {dropped_frames} frames")
        print(f"Percentage of dropped frames: {dropped_percentage:.2f}%")
    else:
        print(f"No dropped frames detected. List len: {expected_frames}, actual frames: {actual_frames}.")

# Example usage
# json_file_path = r"C:\Data\temp_cohort\240917_153225\240917_153243_mtao89-1e\240917_153243_Tracker_data.json"
json_file_path = r"C:\Data\temp_cohort\240917_153225\240917_153243_mtao89-1e\240917_153243_Tracker_data.json"
# json_file_path = r"V:\Behaviour code\2407_July_WT_cohort\Data\240724_115532\240724_115623_wtjx300-6a\240724_115623_Tracker_data.json"
# json_file_path = r"V:\Behaviour code\2409_September_cohort\Data\240919_210634\240919_210653_mtao89-1d\240919_210653_Tracker_data.json"
check_dropped_frames(json_file_path)
