import json
from pathlib import Path
import cv2 as cv
import numpy as np
import re
import traceback
from datetime import datetime

from pynwb import NWBHDF5IO


class Cohort_folder:
    def __init__(
        self,
        cohort_directory,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False
    ):
        """
        :param cohort_directory: Base directory for your data
        :param multi: Whether the data is split across subfolders (multiple mice) or not
        :param portable_data: Whether to use the 'portable_data' logic vs. full raw-data logic
        :param OEAB_legacy: Whether to look for legacy OEAB folder structures
        :param ignore_tests: Skip any session folders that look like test sessions
        :param use_existing_cohort_info: If True and cohort_info.json exists, load from it and skip scanning
        """
        print("Loading cohort info...")
        self.cohort_directory = Path(cohort_directory)
        self.multi = multi
        self.portable_data = portable_data
        self.OEAB_legacy = OEAB_legacy
        self.ignore_tests = ignore_tests

        if not self.cohort_directory.exists():
            raise Exception(f"Folder {self.cohort_directory} does not exist")

        self.json_filename = self.cohort_directory / "cohort_info.json"

        # Attempt loading an existing JSON if requested
        self.cohort = {}
        loaded_from_file = False
        if use_existing_cohort_info and self.json_filename.exists():
            try:
                with open(self.json_filename, 'r') as f:
                    self.cohort = json.load(f)
                print("Loaded cohort info from existing JSON file.")
                loaded_from_file = True
            except Exception as e:
                print(f"Failed to load existing cohort info. Reason: {e}")
                loaded_from_file = False

        # If not loaded from file, build from scratch
        if not loaded_from_file:
            if self.portable_data:
                self.build_portable_data_cohort()
            else:
                self.build_raw_data_cohort()

            # Save the result as JSON
            with open(self.json_filename, 'w') as f:
                json.dump(self.cohort, f, indent=4, sort_keys=True)

        print("Cohort info loaded.\n")

    def get_session(self, session_id):
        """
        Returns the dictionary for the given session_id, or None if not found.
        """
        for mouse_id, mouse_data in self.cohort["mice"].items():
            if session_id in mouse_data["sessions"]:
                return mouse_data["sessions"][session_id]
        return None

    def build_portable_data_cohort(self):
        """
        Build a minimal dictionary for 'portable_data' type folders.
        """
        self.find_mice()   # Sets up self.cohort = {"Cohort name": ..., "mice": {...}}
        self.find_nwbs()   # Finds NWB files and extracts minimal session metadata

        # Add the truncated_start_report check
        self.check_truncated_start_report()

    def build_raw_data_cohort(self):
        """
        Build a minimal dictionary for raw (non-portable) data,
        checking for essential raw data files and NWB/JSON-based metadata.
        """
        self.find_mice()
        self.check_raw_data()
        self.check_truncated_start_report()

    def find_mice(self):
        """
        Populates self.cohort with a top-level "mice" dict containing session directories.
        """
        if not self.multi:
            # Single folder, each subfolder is a session
            self.session_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if len(folder.name) > 13
                and folder.name[13] == "_"
                and folder.is_dir()
                and 'OEAB_recording' not in folder.name
                and (not self.ignore_tests or not self.is_test_session(folder.name))
            ]
        else:
            # Multiple mice subfolders, each containing session subfolders
            self.multi_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
            self.session_folders = []
            for folder in self.multi_folders:
                self.session_folders += [
                    subfolder for subfolder in folder.glob('*')
                    if subfolder.is_dir()
                    and len(subfolder.name) > 13
                    and subfolder.name[13] == "_"
                    and (not self.ignore_tests or not self.is_test_session(subfolder.name))
                ]

        cohort = {"Cohort name": self.cohort_directory.name, "mice": {}}
        for session_folder in self.session_folders:
            mouse_ID = session_folder.name[14:]  # everything after "YYYYMMDD_HHMMSS_"
            if mouse_ID not in cohort["mice"]:
                cohort["mice"][mouse_ID] = {"sessions": {}}
            session_ID = session_folder.name
            cohort["mice"][mouse_ID]["sessions"][session_ID] = {
                "directory": str(session_folder),
                "mouse_id": mouse_ID,
                "session_id": session_ID
            }
        self.cohort = cohort

    def find_nwbs(self):
        """
        For each session (portable data), locate NWB, parse minimal metadata, store in self.cohort.
        """
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                sdict = self.cohort["mice"][mouse]["sessions"][session]
                session_folder = Path(sdict["directory"])

                # Locate NWB
                nwb_path = self.find_file(session_folder, '.nwb')
                nwb_file = str(nwb_path) if nwb_path else "None"
                sdict["NWB_file"] = nwb_file

                # Retrieve metadata if NWB found
                meta = self.get_session_metadata(nwb_file) if nwb_file != "None" else {}
                sdict["phase"] = meta.get('phase', None)
                sdict["cue_duration"] = meta.get('cue_duration', None)
                sdict["wait_duration"] = meta.get('wait_duration', "0")

    def check_raw_data(self):
        """
        For each session, checks if certain files are present (raw video, behavior data, etc.).
        Extracts minimal session metadata. This is for non-portable data.
        """
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                sdict = self.cohort["mice"][mouse]["sessions"][session]
                session_folder = Path(sdict["directory"])
                raw_data = {}

                # Look for required raw data files
                raw_data["raw_video"] = str(self.find_file(session_folder, '.avi'))
                raw_data["behaviour_data"] = str(self.find_file(session_folder, 'behaviour_data'))
                raw_data["tracker_data"] = str(self.find_file(session_folder, 'Tracker_data'))

                if self.OEAB_legacy:
                    raw_data["arduino_DAQ_json"] = str(self.find_file(session_folder, 'ArduinoDAQ.json'))
                    raw_data["OEAB_dir"] = str(self.find_OEAB_dir(session_folder, mouse))
                else:
                    raw_data["arduino_DAQ_h5"] = str(self.find_file(session_folder, 'ArduinoDAQ.h5'))

                # Check if minimal required files exist
                check_list = [raw_data["raw_video"], raw_data["behaviour_data"], raw_data["tracker_data"]]
                if self.OEAB_legacy:
                    check_list.append(raw_data["arduino_DAQ_json"])
                    check_list.append(raw_data["OEAB_dir"])
                else:
                    check_list.append(raw_data["arduino_DAQ_h5"])

                raw_data["is_all_raw_data_present?"] = ("None" not in check_list)

                # Video length from tracker data
                raw_data["video_length"] = self.get_video_length(raw_data["tracker_data"])

                # Session metadata from JSON (behaviour_data) or NWB if present
                if raw_data["behaviour_data"] != "None":
                    meta = self.get_session_metadata(raw_data["behaviour_data"])
                else:
                    meta = {}

                # Merge into the session-level dictionary
                sdict["raw_data"] = raw_data
                sdict["phase"] = meta.get("phase", None)
                sdict["total_trials"] = meta.get("total_trials", None)
                sdict["cue_duration"] = meta.get('cue_duration', None)
                sdict["wait_duration"] = meta.get('wait_duration', "0")

    def check_truncated_start_report(self):
        """
        Adds a boolean key for each session indicating if a subfolder named 'truncated_start_report' exists.
        """
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                sdict = self.cohort["mice"][mouse]["sessions"][session]
                session_path = Path(sdict["directory"])
                report_folder = session_path / "truncated_start_report"
                sdict["has_truncated_start_report"] = report_folder.is_dir()

    def get_session_metadata(self, nwb_or_json_file):
        """
        Retrieves session metadata (phase, total trials, etc.) from a JSON or NWB file.
        """
        if not nwb_or_json_file or nwb_or_json_file == "None":
            return {}
        suffix = Path(nwb_or_json_file).suffix
        session_metadata = {}
        try:
            if suffix == '.json':
                with open(nwb_or_json_file) as f:
                    data = json.load(f)
                session_metadata["phase"] = data.get("Behaviour phase")
                session_metadata["total_trials"] = data.get("Total trials")
                session_metadata["cue_duration"] = data.get("Cue duration", None)
                session_metadata["wait_duration"] = data.get("Wait duration", "0")
            elif suffix == '.nwb':
                with NWBHDF5IO(str(nwb_or_json_file), 'r') as io:
                    nwbfile = io.read()
                    exp_description = nwbfile.experiment_description or ""
                    metadata_parts = {}
                    for part in exp_description.split(';'):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            key, val = part.split(':', 1)
                            metadata_parts[key.strip()] = val.strip()
                        except ValueError:
                            continue
                    session_metadata['phase'] = metadata_parts.get('phase', '')
                    session_metadata['cue_duration'] = metadata_parts.get('cue', '')
                    session_metadata['wait_duration'] = metadata_parts.get('wait', '0')
                    session_metadata["total_trials"] = None
        except Exception as e:
            print(f"Error processing metadata from {nwb_or_json_file}: {str(e)}")
            traceback.print_exc()
        return session_metadata

    def find_OEAB_dir(self, directory, mouse):
        """
        Finds OEAB data by looking for a folder with 'OEAB_recording'.
        If not found, falls back to older heuristic for non-alphabetic folder names.
        """
        if not self.multi:
            for file in directory.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            for file in directory.glob('*'):
                if file.is_dir() and not re.search('[a-zA-Z]', file.name):
                    return file
            return "None"
        else:
            parent = directory.parent
            for file in parent.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            for file in parent.glob('*'):
                if file.is_dir() and (not re.search('[a-zA-Z]', file.name) or file.name == 'New folder'):
                    return file
            return "None"

    def get_video_length(self, tracker_data):
        """
        Parse tracker_data JSON to get rough video length in minutes.
        """
        if tracker_data and tracker_data != "None":
            with open(tracker_data) as f:
                try:
                    data = json.load(f)
                    start_str = data["start_time"]
                    end_str = data["end_time"]
                    time_format = "%y%m%d_%H%M%S"
                    start = datetime.strptime(start_str, time_format)
                    end = datetime.strptime(end_str, time_format)
                    duration = (end - start).total_seconds() / 60
                    return round(duration)
                except ValueError:
                    return None
        return None

    def is_test_session(self, folder_name):
        """
        Check if a session folder name suggests it's a test session
        by examining the portion of the folder name after the underscore at position 13.
        """
        if len(folder_name) > 13 and folder_name[13] == "_":
            mouse_id = folder_name[14:].lower()
            return 'test' in mouse_id
        return False

    def find_file(self, directory, tag):
        """
        Find a file that has 'tag' in its name in the given directory.
        Return the first match or None if not found.
        """
        for file in directory.glob('*'):
            if tag in file.name:
                return file
        return None


def main():
    # Example usage:
    test_dir = Path(r"D:\Behaviour\July_cohort_24\Portable_data")

    cohort = Cohort_folder(
        test_dir,
        multi=True,
        portable_data=True,
        use_existing_cohort_info=False
    )

    # Print result in-memory (cohort info is also saved to JSON)
    import pprint
    pprint.pprint(cohort.cohort)


if __name__ == "__main__":
    main()
