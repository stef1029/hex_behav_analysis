import sys
import json
import cv2
from pathlib import Path

from debug_cohort_folder import Cohort_folder

# Global variable to capture the point from the mouse callback
CLICKED_POINT = None

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback for a single click.
    If user left-clicks, we record that point and close the window.
    """
    global CLICKED_POINT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICKED_POINT = (x, y)
        cv2.destroyAllWindows()  # Stop UI immediately after the click

def select_point_on_image(window_name, image, window_width=1280, window_height=720):
    """
    Show 'image' in a window named window_name with the given size (window_width x window_height).
    The user can:
      - Press ESC to exit the script entirely (returns "ESC").
      - Press 'c' to skip labeling this session (returns None).
      - Left-click once to record that point (returns (x, y)).
    
    Returns:
        "ESC": If the user pressed ESC or closed the window
        None: If the user pressed 'c' to skip labeling
        (x, y): A tuple indicating the clicked coordinates
    """
    global CLICKED_POINT
    CLICKED_POINT = None  # Reset before showing

    # Create a resizable window and then explicitly set its size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # Assign our mouse callback
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF

        # If the user pressed ESC (27), return "ESC" => end the entire script
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return "ESC"

        # If the user pressed 'c', skip labeling for this session
        elif key == ord('c'):
            cv2.destroyAllWindows()
            return None

        # If the user clicked, on_mouse sets CLICKED_POINT
        if CLICKED_POINT is not None:
            break

        # If the user closes the window via the "X" button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            # We treat that like ESC
            return "ESC"

    return CLICKED_POINT

def label_frames_in_cohort(cohort_directory, overwrite=False, window_w=1280, window_h=720):
    """
    Main function:
      1) Instantiate Cohort_folder for the given directory.
      2) For each session with truncated_start_report:
         - Look for <session_id>_middle_frame.png
         - If <session_id>_roi.json exists and overwrite=False, skip
         - Otherwise, open the image for user to:
            * Press ESC => exit entire script
            * Press 'c' => skip labeling this session
            * Left-click => record a single (x,y) point
         - Save that point to <session_id>_roi.json
    """
    cohort = Cohort_folder(
        cohort_directory=cohort_directory,
        multi=True,          # or False, depending on data structure
        portable_data=False, # or True, if using portable data
        OEAB_legacy=True,    # adjust as needed
        ignore_tests=True,
        use_existing_cohort_info=False
    )

    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if not sdict.get("has_truncated_start_report", False):
                # Skip sessions that don't have truncated_start_report
                continue

            session_folder = Path(sdict["directory"])
            tsr_folder = session_folder / "truncated_start_report"

            # The middle-frame image is <session_id>_middle_frame.png
            middle_frame_path = tsr_folder / f"{session_id}_middle_frame.png"
            if not middle_frame_path.is_file():
                print(f"No middle frame found for session {session_id} at {middle_frame_path}. Skipping.")
                continue

            # ROI JSON path
            roi_json_path = tsr_folder / f"{session_id}_roi.json"
            if roi_json_path.is_file() and not overwrite:
                # ROI already exists, and we are NOT overwriting
                print(f"ROI file already exists for session {session_id} ({roi_json_path}). Skipping.")
                continue

            # Read the image
            image = cv2.imread(str(middle_frame_path))
            if image is None:
                print(f"Failed to read the image at {middle_frame_path}. Skipping.")
                continue

            print(f"\n=== Session: {session_id} ===")
            print("Commands:")
            print(" - Left-click: record a single (x,y) point for ROI.")
            print(" - ESC: exit the entire script immediately.")
            print(" - c: skip this session (no ROI saved), move on to next.")

            result = select_point_on_image(
                window_name="Select Point",
                image=image,
                window_width=window_w,
                window_height=window_h
            )

            if result == "ESC":
                print("User pressed ESC or closed window. Exiting entire script.")
                sys.exit(0)

            if result is None:
                print(f"Skipping ROI labeling for session {session_id}.")
                continue

            # Otherwise, user clicked => we have an (x, y)
            x, y = result
            roi_data = {
                "session_id": session_id,
                "x": int(x),
                "y": int(y)
            }

            # Save to JSON
            try:
                with open(roi_json_path, "w") as f:
                    json.dump(roi_data, f, indent=2)
                print(f"Saved ROI (single point) for session {session_id} to {roi_json_path}")
            except Exception as e:
                print(f"Error saving ROI data to {roi_json_path}: {e}")

def main():
    """
    Usage:
        python label_frames_cohort.py <cohort_directory>
    Example:
        python label_frames_cohort.py /data/MyCohort
    """

    cohort_directory = r"Z:\debug_vids\September_cohort_label_frames"
    overwrite = False
    label_frames_in_cohort(cohort_directory, overwrite=overwrite, window_w=1280, window_h=1024)


if __name__ == "__main__":
    main()
