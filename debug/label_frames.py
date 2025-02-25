import sys
import json
import cv2
from pathlib import Path

from debug_cohort_folder import Cohort_folder
# ^ Adjust this import to match where your Cohort_folder is defined.

def label_frames_in_cohort(cohort_directory):
    """
    1) Instantiate Cohort_folder for the given directory.
    2) For each session that has truncated_start_report:
       - Find the extracted middle-frame image
       - Prompt user to select ROI via cv2.selectROI
       - Save the ROI coordinates in a JSON file: <session_id>_roi.json
    """

    # 1) Load or build the cohort structure
    cohort = Cohort_folder(
        cohort_directory=cohort_directory,
        multi=True,          # Or False, depending on your data organization
        portable_data=False, # Or True, if using "portable_data" style
        OEAB_legacy=True,    # Depends on your folder layout
        ignore_tests=True,
        use_existing_cohort_info=False
    )

    cohort_path = Path(cohort_directory).resolve()

    # 2) Iterate over all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if not sdict.get("has_truncated_start_report", False):
                # Skip sessions that don't have the truncated_start_report folder
                continue

            session_folder = Path(sdict["directory"])
            tsr_folder = session_folder / "truncated_start_report"

            # The middle-frame image might be named <session_id>_middle_frame.png
            middle_frame_path = tsr_folder / f"{session_id}_middle_frame.png"
            if not middle_frame_path.is_file():
                print(f"No middle frame found for session {session_id} at {middle_frame_path}, skipping.")
                continue

            # Check if we already have ROI data. If so, we can skip or allow re-label:
            roi_json_path = tsr_folder / f"{session_id}_roi.json"
            if roi_json_path.is_file():
                print(f"ROI file already exists for session {session_id} at {roi_json_path}, skipping.")
                continue

            # Load the image
            image = cv2.imread(str(middle_frame_path))
            if image is None:
                print(f"Failed to read the image at {middle_frame_path}, skipping.")
                continue

            # Let the user select ROI
            print(f"\nSession: {session_id}\nSelect ROI in the displayed image.")
            roi = cv2.selectROI("Select ROI", image, showCrosshair=True)
            cv2.destroyWindow("Select ROI")

            x, y, w, h = roi
            if w == 0 or h == 0:
                print("No valid ROI selected or user canceled. Skipping.")
                continue

            # Save the ROI data to JSON
            roi_data = {
                "session_id": session_id,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
            try:
                with open(roi_json_path, "w") as f:
                    json.dump(roi_data, f, indent=2)
                print(f"Saved ROI to {roi_json_path}")
            except Exception as e:
                print(f"Error saving ROI data to {roi_json_path}: {e}")


def main():
    """
    Usage:
        python label_frames_cohort.py <cohort_directory>
    Example:
        python label_frames_cohort.py /data/MyCohort
    """
    if len(sys.argv) < 2:
        print("Usage: python label_frames_cohort.py <cohort_directory>")
        sys.exit(1)

    cohort_directory = sys.argv[1]
    label_frames_in_cohort(cohort_directory)


if __name__ == "__main__":
    main()
