import cv2
import numpy as np

def label_sensor_touches(video_path, brightness_threshold=150.0):
    """
    Loads a video, asks the user to select a region of interest (ROI) on the first frame,
    and calculates the average brightness of that ROI on each frame. Returns a list
    of timestamps (in milliseconds) when the brightness crosses the specified threshold.
    
    :param video_path: Path to the video file.
    :param brightness_threshold: Brightness threshold for detecting a 'touch' event.
    :return: List of timestamps (in milliseconds) where brightness crosses the threshold.
    """

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot read the first frame of the video.")

    # Let the user select ROI on the first frame
    # The function returns a tuple (x, y, w, h)
    roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True)
    cv2.destroyWindow("Select ROI")  # close the ROI selection window

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise ValueError("No ROI selected or invalid ROI dimensions.")

    # Prepare to loop over frames
    timestamps = []
    was_above_threshold = False

    # We want to find when brightness crosses the threshold: specifically
    # from below-threshold to above-threshold. If you want all frames above,
    # or crossing in both directions, adjust the logic accordingly.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current video timestamp in milliseconds
        # (You can also retrieve frame index and compute your own if needed)
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Extract ROI, convert to grayscale
        roi_frame = frame[y : y + h, x : x + w]
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Compute average brightness
        avg_brightness = np.mean(gray_roi)

        # Check threshold crossing (from below to above)
        if (avg_brightness >= brightness_threshold) and (not was_above_threshold):
            timestamps.append(current_time_ms)
            was_above_threshold = True
        elif avg_brightness < brightness_threshold and was_above_threshold:
            was_above_threshold = False

    cap.release()

    return timestamps


if __name__ == "__main__":
    video_file = "/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE/250121_142325/250121_142326_wtjp271-5d/250121_174347_output.avi"
    threshold_value = 150.0  # adjust as needed

    crossing_times = label_sensor_touches(video_file, threshold_value)

    print("Brightness crossing timestamps (ms):")
    for t in crossing_times:
        print(t)
