import json
from pathlib import Path
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
import traceback
from datetime import datetime

from pynwb import NWBHDF5IO

from hex_behav_analysis.utils.plot_graphical_cohort_info import graphical_cohort_info


class Cohort_folder:
    def __init__(
        self,
        cohort_directory,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False,
        plot=False
    ):
        """
        :param cohort_directory: Base directory for your data
        :param multi: Whether the data is split across subfolders (multiple mice) or not
        :param portable_data: Whether to use the 'portable_data' logic vs. full raw-data logic
        :param OEAB_legacy: Whether to look for legacy OEAB folder structures
        :param ignore_tests: Skip any session folders that look like test sessions
        :param use_existing_cohort_info: If True and cohort_info.json exists, load from it and skip scanning
        :param plot: Whether to produce a cohort summary plot (if relevant)
        """
        print("Loading cohort info...")
        self.cohort_directory = Path(cohort_directory)
        self.multi = multi
        self.portable_data = portable_data
        self.OEAB_legacy = OEAB_legacy
        self.ignore_tests = ignore_tests
        self.plot = plot

        if not self.cohort_directory.exists():
            raise Exception(f"Folder {self.cohort_directory} does not exist")

        self.json_filename = self.cohort_directory / "cohort_info.json"
        self.concise_json_filename = self.cohort_directory / "concise_cohort_info.json"

        # Try loading existing cohort_info.json if use_existing_cohort_info is True
        loaded_from_file = False
        if use_existing_cohort_info and self.json_filename.exists():
            try:
                with open(self.json_filename, 'r') as f:
                    self.cohort = json.load(f)
                if self.concise_json_filename.exists():
                    with open(self.concise_json_filename, 'r') as f:
                        self.cohort_concise = json.load(f)
                else:
                    # If concise doesn't exist, create an empty placeholder or skip
                    self.cohort_concise = {}
                print("Loaded cohort info from existing JSON files.")
                loaded_from_file = True
            except Exception as e:
                print(f"Failed to load existing cohort info. Reason: {e}")
                loaded_from_file = False

        # If we did NOT successfully load from file, do the usual scanning/parsing
        if not loaded_from_file:
            if self.portable_data:
                self.init_portable_data()
            else:
                self.init_raw_data()

            # If you want the automatic plotting after building from scratch:
            if self.plot:
                self.plot_graphical_cohort_info()

        print("Cohort info loaded.\n")


    def plot_graphical_cohort_info(self, show=False):
        graphical_cohort_info(self.cohort, self.cohort_directory, show)


    def is_test_session(self, folder_name):
        """
        Check if a session folder represents a test session by examining
        the portion of the folder name after the underscore at position 13.
        """
        if len(folder_name) > 13 and folder_name[13] == "_":
            mouse_id = folder_name[14:].lower()
            return 'test' in mouse_id
        return False


    def init_portable_data(self):
        # make initial dictionary and mouse sub dictionaries with prelimiary information.
        self.find_mice()

        # get nwb file location and add to each session dictionary
        self.find_nwbs()

        # save as json
        with open(self.json_filename, 'w') as f:
            json.dump(self.cohort, f, indent=4, sort_keys=True)


    def find_nwbs(self):
        """
        Finds .nwb files for each session, parses experiment_description
        to get phase, cue, and wait durations, and stores in self.cohort.
        """
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])

                # 1) Locate NWB file
                nwb_path = self.find_file(session_folder, '.nwb')
                if nwb_path is not None:
                    nwb_file = str(nwb_path)
                else:
                    nwb_file = "None"
                    print(f"[WARNING] No NWB file found for session {session_folder}")

                # 2) Store NWB path
                self.cohort["mice"][mouse]["sessions"][session]["NWB_file"] = nwb_file

                # 3) Retrieve metadata (phase, cue_duration, wait_duration)
                meta = self.get_session_metadata(nwb_file) if nwb_file != "None" else {}

                phase = meta.get('phase', None)
                cue_duration = meta.get('cue_duration', None)
                wait_duration = meta.get('wait_duration', "0")

                # 4) Store in dict
                self.cohort["mice"][mouse]["sessions"][session]["Behaviour_phase"] = phase
                self.cohort["mice"][mouse]["sessions"][session]["cue_duration"] = cue_duration
                self.cohort["mice"][mouse]["sessions"][session]["wait_duration"] = wait_duration
                self.cohort["mice"][mouse]["sessions"][session]["portable"] = True


    def init_raw_data(self):
        self.find_mice()
        self.check_raw_data()
        self.check_for_preliminary_analysis()

        # save as json
        with open(self.json_filename, 'w') as f:
            json.dump(self.cohort, f, indent=4, sort_keys=True)

        # save concise logs as json
        self.make_concise_training_logs()
        with open(self.concise_json_filename, 'w') as f:
            json.dump(self.cohort_concise, f, indent=4, sort_keys=True)

        if self.plot:
            self.plot_graphical_cohort_info()

        print('Cohort info loaded')


    def get_session(self, ID, concise=False):
        """
        Takes a session ID and returns a dictionary containing the info
        from cohort info about that session.
        """
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                if session == ID:
                    if not concise:
                        return self.cohort["mice"][mouse]["sessions"][session]
                    else:
                        return self.cohort_concise["complete_data"].get(mouse, {}).get(session, None)
        return None


    def phases(self):
        """
        Makes it easy to see the sessions for each known phase in a cohort.
        """
        phases = ["1", "2", "3", "3b", "3c", "4", "4b", "4c", "test", 
                "5", "6", "7", "8", "9", "9b", "9c", "10"]
        phase_dict = {phase: {} for phase in phases}
        if not self.portable_data:
            for mouse in self.cohort_concise.get("complete_data", {}):
                for session in self.cohort_concise["complete_data"][mouse]:
                    phase = self.cohort_concise["complete_data"][mouse][session]["Behaviour_phase"]
                    session_path = self.cohort["mice"][mouse]["sessions"][session]["directory"]
                    try:
                        phase_dict[phase][session] = {
                            "path": session_path,
                            "total_trials": self.cohort_concise["complete_data"][mouse][session]["total_trials"],
                            "video_length": self.cohort_concise["complete_data"][mouse][session]["video_length"],
                            "mouse": mouse
                        }
                    except KeyError:
                        print(f"KeyError when trying to assign phase '{phase}' for session '{session}'.")
                        print("Check if 'phase' in your data matches one of:", phases)
                        raise
        else:
            for mouse in self.cohort["mice"]:
                for session in self.cohort["mice"][mouse]["sessions"]:
                    phase = self.cohort["mice"][mouse]["sessions"][session]["Behaviour_phase"]
                    session_path = self.cohort["mice"][mouse]["sessions"][session]["directory"]
                    try:
                        phase_dict[phase][session] = {"path": session_path, "mouse": mouse}
                    except KeyError:
                        print(f"KeyError when trying to assign phase '{phase}' for session '{session}'.")
                        print("Check if 'phase' in your data matches one of:", phases)
                        raise

        return phase_dict


    def make_concise_training_logs(self):
        """
        Creates a concise summary dictionary self.cohort_concise for easy overview.
        Splits sessions into complete_data / incomplete_data keys.
        """
        cohort_info = self.cohort
        self.cohort_concise = {}

        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                try:
                    # Check if session has valid raw data
                    is_data_present = cohort_info["mice"][mouse]["sessions"][session]["raw_data"].get("is_all_raw_data_present?", False)
                    
                    # Get metadata
                    try:
                        behaviour_phase = cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["session_metadata"].get("phase")
                        total_trials = cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["session_metadata"].get("total_trials")
                        video_length = cohort_info["mice"][mouse]["sessions"][session]["raw_data"].get("video_length")
                    except (KeyError, TypeError):
                        # Handle case where session_metadata might be missing or incomplete
                        behaviour_phase = None
                        total_trials = None
                        video_length = None
                        is_data_present = False
                    
                    # Add session to appropriate section
                    if is_data_present:
                        if "complete_data" not in self.cohort_concise:
                            self.cohort_concise["complete_data"] = {}
                        if mouse not in self.cohort_concise["complete_data"]:
                            self.cohort_concise["complete_data"][mouse] = {}
                        self.cohort_concise["complete_data"][mouse][session] = {
                            "Behaviour_phase": behaviour_phase,
                            "total_trials": total_trials,
                            "video_length": video_length,
                            "mouse_id": mouse,
                        }
                    else:
                        if "incomplete_data" not in self.cohort_concise:
                            self.cohort_concise["incomplete_data"] = {}
                        if mouse not in self.cohort_concise["incomplete_data"]:
                            self.cohort_concise["incomplete_data"][mouse] = {}
                        self.cohort_concise["incomplete_data"][mouse][session] = {
                            "Behaviour_phase": behaviour_phase,
                            "total_trials": total_trials,
                            "video_length": video_length,
                            "mouse_id": mouse,
                        }
                except Exception as e:
                    # Check if session is in the ignore list first
                    ignore = [
                        "240807_163610_wtjp254-4b",  # forgot to turn scales on
                        "240725_110604_wtjx300-6a",  # no trials
                        "240720_115745_wtjx300-6a",  # no trials
                        "240731_120318_wtjx300-6a",
                        "240720_114019_wtjx300-6a",
                        "240720_120336_wtjx300-6a",
                        "240720_120413_wtjx300-6a",
                        "240725_110604_wtjx300-6b",  # no trials
                        "240731_120318_wtjx300-6b",
                        "240720_114019_wtjx300-6b",
                        "240720_120336_wtjx300-6b",
                        "240720_120413_wtjx300-6b",
                        "240719_150822_wtjx261-2a",
                        "240716_141151_wtjx261-2a",
                        "240719_150822_wtjx307-6b",
                        # Add the problematic session ID here if known
                        "250401_170043_mtao106-3e"
                    ]
                    if session in ignore:
                        continue
                    else:
                        print(f"Error processing {session}: {str(e)}")
                    continue


    def text_summary_cohort_info(self):
        """
        Example textual summary. Requires self.cohort_concise["complete_data"] loaded.
        """
        if "complete_data" not in self.cohort_concise:
            print("No complete_data in cohort_concise. Please generate or load it first.")
            return

        data = pd.DataFrame(self.cohort_concise["complete_data"])
        data.reset_index(inplace=True)
        data.rename(columns={"index": "SessionID"}, inplace=True)

        data['SessionDate'] = pd.to_datetime(data['SessionID'].str[:6], format='%y%m%d').dt.date
        data_melted = data.melt(id_vars=["SessionID", "SessionDate"], var_name="Mouse", value_name="Details")
        data_melted.dropna(subset=["Details"], inplace=True)
        data_melted["BehaviourPhase"] = data_melted["Details"].apply(lambda x: x.get('Behaviour_phase'))
        data_melted["TotalTrials"] = data_melted["Details"].apply(lambda x: x.get('total_trials'))
        data_melted["VideoLength"] = data_melted["Details"].apply(lambda x: x.get('video_length'))

        grouped = data_melted.groupby('SessionDate')
        summary_lines = []
        for session_date, group in grouped:
            mice = group['Mouse'].unique()
            total_trials = group.groupby('Mouse')['TotalTrials'].sum()
            behavior_phases = group.groupby('Mouse')['BehaviourPhase'].apply(lambda x: ', '.join(x.unique()))
            session_counts = group.groupby('Mouse')['SessionID'].nunique()
            video_lengths = group.groupby('Mouse')['VideoLength'].apply(lambda x: ', '.join(x.unique()))

            summary_lines.append(f"Date: {session_date}")
            for mouse in mice:
                summary_lines.append(f"  Mouse: {mouse}")
                summary_lines.append(f"    Total Trials: {total_trials[mouse]}")
                summary_lines.append(f"    Behavior Phases: {behavior_phases[mouse]}")
                summary_lines.append(f"    Number of Sessions: {session_counts[mouse]}")
                summary_lines.append(f"    Video Lengths: {video_lengths[mouse]}")
            summary_lines.append("")

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        filename = self.cohort_directory / "cohort_summary.txt"
        with open(filename, 'w') as file:
            file.write(summary_text)


    def find_mice(self):
        """
        Populates self.cohort = {...} with a top-level "mice" dict.
        """
        if not self.multi:
            self.session_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if len(folder.name) > 13
                and folder.name[13] == "_"
                and folder.is_dir()
                and 'OEAB_recording' not in folder.name
                and (not self.ignore_tests or not self.is_test_session(folder.name))
            ]
        else:
            self.multi_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
            self.session_folders = [
                subfolder for folder in self.multi_folders 
                for subfolder in folder.glob('*')
                if subfolder.is_dir() 
                and len(subfolder.name) > 13
                and subfolder.name[13] == "_"
                and 'OEAB_recording' not in subfolder.name
                and (not self.ignore_tests or not self.is_test_session(subfolder.name))
            ]

        # get current date and time as string:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Current date: {current_date}")
        cohort = {"Cohort name": self.cohort_directory.name, "Time refreshed": current_date, "mice": {}}
        for session_folder in self.session_folders:
            mouse_ID = session_folder.name[14:]  # everything after "YYYYMMDD_HHMMSS_"
            if mouse_ID not in cohort["mice"]:
                cohort["mice"][mouse_ID] = {"sessions": {}}
            session_ID = session_folder.name
            cohort["mice"][mouse_ID]["sessions"][session_ID] = {
                "directory": str(session_folder),
                "mouse_id": mouse_ID,
                "session_id": session_ID,
                "portable": False
            }
        self.cohort = cohort


    def check_raw_data(self):
        """
        For each session, checks if the expected raw data files are present.
        Also checks if the 'phase' in the session metadata is one of the valid
        phases (exact string match). If not, we tag the session as incomplete.
        """
        # Define which phases are considered valid (in exact, lower-case form)
        valid_phases = ["1", "2", "3", "3b", "3c", "4", "4b", "4c",
                        "test", "5", "6", "7", "8", "9", "9b", "9c", "10"]

        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])
                raw_data = {}
                check_list = []

                # Look for required raw data files
                raw_data["raw_video"] = str(self.find_file(session_folder, '.avi'))
                raw_data["behaviour_data"] = str(self.find_file(session_folder, 'behaviour_data'))
                check_list.append(raw_data["behaviour_data"])
                raw_data["tracker_data"] = str(self.find_file(session_folder, 'Tracker_data'))
                check_list.append(raw_data["tracker_data"])

                if self.OEAB_legacy:
                    raw_data["arduino_DAQ_json"] = str(self.find_file(session_folder, 'ArduinoDAQ.json'))
                    check_list.append(raw_data["arduino_DAQ_json"])
                    raw_data["OEAB"] = str(self.find_OEAB_dir(session_folder, mouse))
                    check_list.append(raw_data["OEAB"])
                else:
                    raw_data["arduino_DAQ_h5"] = str(self.find_file(session_folder, 'ArduinoDAQ.h5'))
                    check_list.append(raw_data["arduino_DAQ_h5"])

                # Check for video fallback (temp bmp frames)
                bmp_file_check = self.find_file(session_folder, 'temp') is not None
                video_check = "None" if (raw_data["raw_video"] == "None" and not bmp_file_check) else True
                check_list.append(video_check)

                # Load behaviour data to check for scales
                if raw_data["behaviour_data"] != "None":
                    try:
                        with open(raw_data["behaviour_data"]) as f:
                            try:
                                data = json.load(f)
                                raw_data["scales_data"] = len(data["Scales data"]) > 0
                                check_list.append(raw_data["scales_data"])
                            except (json.JSONDecodeError, KeyError) as json_err:
                                print(f"JSON error in {session}: {str(json_err)}")
                                raw_data["scales_data"] = False
                                check_list.append("None")  # Mark as missing
                                # Add to missing files list
                                if "missing_files" not in raw_data:
                                    raw_data["missing_files"] = []
                                raw_data["missing_files"].append("invalid_behaviour_data_json")
                    except Exception as e:
                        print(f"Error reading behaviour data for {session}: {str(e)}")
                        raw_data["scales_data"] = False
                        check_list.append("None")
                else:
                    raw_data["scales_data"] = False
                    check_list.append("None")

                # Determine if all raw files are present (before phase check)
                if "None" not in check_list:
                    raw_data["is_all_raw_data_present?"] = True
                else:
                    raw_data["is_all_raw_data_present?"] = False

                # Track which files are missing
                if "missing_files" not in raw_data:
                    raw_data["missing_files"] = []
                    
                if raw_data["raw_video"] == "None":
                    raw_data["missing_files"].append("raw_video")
                if raw_data["behaviour_data"] == "None":
                    raw_data["missing_files"].append("behaviour_data")
                if raw_data["tracker_data"] == "None":
                    raw_data["missing_files"].append("tracker_data")

                if self.OEAB_legacy:
                    if raw_data["arduino_DAQ_json"] == "None":
                        raw_data["missing_files"].append("arduino_DAQ_json")
                    if raw_data["OEAB"] == "None":
                        raw_data["missing_files"].append("OEAB")
                else:
                    if raw_data["arduino_DAQ_h5"] == "None":
                        raw_data["missing_files"].append("arduino_DAQ_h5")

                # If OEAB_legacy, store OEAB folder contents
                if self.OEAB_legacy:
                    OEAB_contents = self.get_OEAB_file_info(raw_data["OEAB"])
                    raw_data["OEAB_contents"] = OEAB_contents

                # Get the video length from tracker_data
                raw_data["video_length"] = self.get_video_length(raw_data["tracker_data"])

                # Read the session metadata from the relevant file (JSON or NWB)
                session_metadata = self.get_session_metadata(raw_data["behaviour_data"])
                # Make sure session_metadata is a dictionary, never None
                if session_metadata is None:
                    session_metadata = {
                        "phase": None,
                        "total_trials": None,
                        "cue_duration": None,
                        "wait_duration": "0"
                    }
                raw_data["session_metadata"] = session_metadata

                # Check if the JSON was invalid (new flag from get_session_metadata)
                if session_metadata.get("invalid_json", False) or session_metadata.get("error_occurred", False):
                    raw_data["is_all_raw_data_present?"] = False
                    if "invalid_json" not in raw_data["missing_files"]:
                        raw_data["missing_files"].append("invalid_json")

                # -------------------------
                # Phase validation:
                # -------------------------
                # If the session's phase is not an exact match in valid_phases,
                # force "is_all_raw_data_present?" to be False.
                phase_in_file = session_metadata.get("phase", None)
                if phase_in_file not in valid_phases:
                    raw_data["is_all_raw_data_present?"] = False
                    # Optionally note "invalid_phase" in missing_files for clarity
                    if "invalid_phase" not in raw_data["missing_files"]:
                        raw_data["missing_files"].append("invalid_phase")

                # Store the raw_data dict back into self.cohort
                self.cohort["mice"][mouse]["sessions"][session]["raw_data"] = raw_data


    def get_video_length(self, tracker_data):
        if tracker_data != "None":
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
        else:
            return None


    def get_OEAB_file_info(self, OEAB_file):
        OEAB_contents = {}
        if OEAB_file != "None":
            OEAB_nodes = list(Path(OEAB_file).glob('*'))
            OEAB_contents["nodes"] = [str(node) for node in OEAB_nodes]
        else:
            OEAB_nodes = None
            OEAB_contents["nodes"] = None

        if OEAB_nodes is not None and len(OEAB_nodes) > 0:
            name0 = OEAB_nodes[0].name
            if name0 in ["Record Node 113", "Record Node 101", "Record Node 120"]:
                recording_no = 0
                for file in OEAB_nodes[0].glob('*'):
                    if "ADC1" in file.name:
                        recording_no += 1
                OEAB_contents["recordings"] = recording_no
            else:
                OEAB_contents["recordings"] = None
        else:
            OEAB_contents["recordings"] = None

        return OEAB_contents


    def get_session_metadata(self, nwb_or_json_file):
        session_metadata = {}
        try:
            if nwb_or_json_file != "None":
                suffix = Path(nwb_or_json_file).suffix
                if suffix == '.json':
                    try:
                        with open(nwb_or_json_file) as f:
                            data = json.load(f)
                        session_metadata["phase"] = data.get("Behaviour phase")
                        session_metadata["total_trials"] = data.get("Total trials")
                        session_metadata["cue_duration"] = data.get("Cue duration", None)
                        session_metadata["wait_duration"] = data.get("Wait duration", "0")
                    except json.JSONDecodeError as json_err:
                        print(f"JSONDecodeError for file {nwb_or_json_file}: {str(json_err)}")
                        # Set metadata to default values for invalid JSON
                        session_metadata["phase"] = None
                        session_metadata["total_trials"] = None
                        session_metadata["cue_duration"] = None
                        session_metadata["wait_duration"] = "0"
                        # Add a flag to indicate JSON was invalid
                        session_metadata["invalid_json"] = True
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
                                key = key.strip()
                                val = val.strip()
                                metadata_parts[key] = val
                            except ValueError:
                                print(f"[WARNING] Couldn't parse '{part}' with ':'. Skipping.")
                                continue

                        session_metadata['phase'] = metadata_parts.get('phase', '').strip()
                        session_metadata['cue_duration'] = metadata_parts.get('cue', '').strip()
                        session_metadata['wait_duration'] = metadata_parts.get('wait', '0').strip()
                        session_metadata["total_trials"] = None
            else:
                session_metadata["phase"] = None
                session_metadata["total_trials"] = None
                session_metadata["cue_duration"] = None
                session_metadata["wait_duration"] = "0"
        except Exception as e:
            print(f"Error processing metadata from {nwb_or_json_file}: {str(e)}")
            traceback.print_exc()
            # Return empty dictionary with default values instead of None
            session_metadata = {
                "phase": None,
                "total_trials": None,
                "cue_duration": None,
                "wait_duration": "0",
                "error_occurred": True
            }

        return session_metadata


    def check_for_preliminary_analysis(self):
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])
                processed_data = {}
                check_list = []

                processed_data["sendkey_logs"] = str(self.find_file(session_folder, 'sendkey_logs'))
                check_list.append(processed_data["sendkey_logs"])
                processed_data["video_frametimes"] = str(self.find_file(session_folder, 'video_frame_times'))
                check_list.append(processed_data["video_frametimes"])
                processed_data["sendkey_metadata"] = str(self.find_file(session_folder, 'behaviour_data'))
                check_list.append(processed_data["sendkey_metadata"])
                processed_data["NWB_file"] = str(self.find_file(session_folder, '.nwb'))
                processed_data["DLC"] = self.find_DLC_files(session_folder)

                if self.OEAB_legacy:
                    processed_data["processed_DAQ_data"] = str(self.find_file(session_folder, 'processed_DAQ_data'))
                    check_list.append(processed_data["processed_DAQ_data"])

                if "None" not in check_list:
                    processed_data["preliminary_analysis_done?"] = True
                else:
                    processed_data["preliminary_analysis_done?"] = False

                self.cohort["mice"][mouse]["sessions"][session]["processed_data"] = processed_data


    def find_DLC_files(self, session_folder):
        """
        Find DLC-related files in the session folder using modern naming patterns.
        
        Args:
            session_folder: Path to the session folder containing DLC files
            
        Returns:
            Dictionary containing paths to the labelled video and coordinate files
        """
        DLC_files = {}
        
        # Convert to Path object for easier handling
        session_path = Path(session_folder)
        
        # Find labelled video - look for files with 'labeled' or 'labelled' in the name
        labelled_video = None
        for video_file in session_path.glob("*"):
            if video_file.is_file() and ('labeled' in video_file.name.lower() or 'labelled' in video_file.name.lower()):
                labelled_video = video_file
                break
        
        DLC_files["labelled_video"] = str(labelled_video) if labelled_video else "None"
        
        # Find coordinates CSV - look for CSV files with DLC_Resnet50 pattern
        coords_csv = None
        for csv_file in session_path.glob("*DLC_Resnet50*.csv"):
            if csv_file.is_file():
                coords_csv = csv_file
                break
        
        DLC_files["coords_csv"] = str(coords_csv) if coords_csv else "None"
        
        # Find H5 files with DLC_Resnet50 pattern
        coords_h5 = None
        for h5_file in session_path.glob("*DLC_Resnet50*.h5"):
            if h5_file.is_file():
                coords_h5 = h5_file
                break
        
        DLC_files["coords_h5"] = str(coords_h5) if coords_h5 else "None"
        
        # Find pickle metadata files with DLC_Resnet50 pattern
        meta_pickle = None
        for pickle_file in session_path.glob("*DLC_Resnet50*_meta.pickle"):
            if pickle_file.is_file():
                meta_pickle = pickle_file
                break
        
        DLC_files["meta_pickle"] = str(meta_pickle) if meta_pickle else "None"
        
        return DLC_files


    def find_file(self, directory, tag):
        for file in directory.glob('*'):
            if tag in file.name:
                return file
        return None


    def find_OEAB_dir(self, directory, mouse):
        """
        Finds OEAB data by looking for a folder with 'OEAB_recording'.
        If not found, falls back to older method (folder with no letters).
        """
        if not self.multi:
            for file in directory.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            for file in directory.glob('*'):
                if file.is_dir() and not re.search('[a-zA-Z]', file.name):
                    return file
            return None
        else:
            parent = directory.parent
            for file in parent.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            for file in parent.glob('*'):
                if file.is_dir() and (not re.search('[a-zA-Z]', file.name) or file.name == 'New folder'):
                    return file
            return None


def main():
    # Example usage:
    test_dir = Path(r"D:\Behaviour\July_cohort_24\Portable_data")

    # If you want to rely on scanning:
    # cohort = Cohort_folder(
    #     test_dir,
    #     multi=True,
    #     portable_data=True,
    #     use_existing_cohort_info=False,
    #     plot=True
    # )

    # If you want to load from existing JSON if present:
    cohort = Cohort_folder(
        test_dir,
        multi=True,
        portable_data=True,
        use_existing_cohort_info=True,
        plot=True
    )

    # Possibly do some queries on cohort
    # session_info = cohort.get_session("240311_183300")
    # print(session_info)


if __name__ == "__main__":
    main()
