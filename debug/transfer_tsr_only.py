#!/usr/bin/env python3

import sys
import shutil
import os
from pathlib import Path

from debug_cohort_folder import Cohort_folder
# ^ Adjust this import if your Cohort_folder class is defined elsewhere.

def transfer_tsr_only(cohort_dir, destination_dir):
    """
    1) Builds a Cohort_folder to read all session metadata.
    2) For each session that has a 'truncated_start_report' folder, copies
       ONLY that folder (and its contents) to the destination directory, preserving
       the session's subpath structure.
    """

    # 1) Instantiate the cohort
    cohort = Cohort_folder(
        cohort_directory=cohort_dir,
        multi=True,                # or False, depending on your data structure
        portable_data=False,       # or True, if you're using portable data
        OEAB_legacy=True,          # depends on your folder structure
        ignore_tests=True,         # or False, depending on your use case
        use_existing_cohort_info=False
    )

    cohort_path = Path(cohort_dir).resolve()
    destination_path = Path(destination_dir).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    # 2) Iterate over all mice and sessions
    for mouse_id, mouse_data in cohort.cohort["mice"].items():
        for session_id, sdict in mouse_data["sessions"].items():
            if sdict.get("has_truncated_start_report", False):
                # Found a session that has truncated_start_report
                session_folder = Path(sdict["directory"])
                tsr_source = session_folder / "truncated_start_report"

                # Build the corresponding subpath in the destination
                # so that we preserve the structure from the cohort root.
                try:
                    session_subpath = session_folder.relative_to(cohort_path)
                except ValueError:
                    # If session_folder isn't under cohort_path, you'll need
                    # a different logic. For now, just skip or handle differently.
                    print(f"Session folder {session_folder} is not relative to {cohort_path}, skipping.")
                    continue

                # e.g. if session_subpath is "Mouse1/20230221_140005_mouseA"
                # then the new session path is <destination_dir>/Mouse1/20230221_140005_mouseA
                new_session_path = destination_path / session_subpath
                new_session_path.mkdir(parents=True, exist_ok=True)

                tsr_dest = new_session_path / "truncated_start_report"

                if tsr_source.is_dir():
                    # Copy the entire TSR folder into the new location
                    # Using copytree with dirs_exist_ok=True (Python 3.8+). If on older Python,
                    # you'd have to manually copy contents or handle existence checks.
                    try:
                        shutil.copytree(tsr_source, tsr_dest, dirs_exist_ok=True)
                        print(f"Copied TSR from {tsr_source} to {tsr_dest}")
                    except Exception as e:
                        print(f"Error copying {tsr_source} -> {tsr_dest}: {str(e)}")
                else:
                    print(f"Truncated folder not found or not a directory at {tsr_source}, skipping.")

def main():
    """
    Usage:
        python transfer_tsr_only.py <cohort_directory> <destination_directory>
    Example:
        python transfer_tsr_only.py /data/OldCohort /data/TruncatedReportsOnly
    """

    cohort_dir = r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE"
    destination_dir = r"/cephfs2/srogers/debug_vids/Lynn_label_frames"

    transfer_tsr_only(cohort_dir, destination_dir)

if __name__ == "__main__":
    main()
