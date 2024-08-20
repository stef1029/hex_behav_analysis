import json
from pathlib import Path
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
import traceback

from pynwb import NWBHDF5IO, NWBFile, TimeSeries, ProcessingModule, load_namespaces
from pynwb.behavior import SpatialSeries
from pynwb.file import Subject, LabMetaData
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, export_spec

# this takes the current cohort behaviour folder and manages the data in it. 
# my hope is that it will automatically group the individual mouse files into lists, detect if the correct files are there
# it should also keep track of whether the initial analysis has been done, and if not, give the option to run it.
# i think this might be where I create the high level mouse class.





class Cohort_folder:
    def __init__(self, cohort_directory, multi = True, plot = False, portable_data = False):        # could have a function that detects if it's multi automatically
        print('Loading cohort info...')
        self.cohort_directory = Path(cohort_directory)
        self.multi = multi
        self.plot = plot
        self.portable_data = portable_data

        # check if folder exists:
        if not self.cohort_directory.exists():
            raise Exception(f"Folder {self.cohort_directory} does not exist")

        if self.portable_data == False:
            self.init_raw_data()
        else:
            self.init_portable_data()
    
    def init_portable_data(self):
        # make initial dictionary and mouse sub dictionaries with prelimiary information.
        self.find_mice()

        # get nwb file location and add to each session dictionary
        self.find_nwbs()
        # save as json
        self.json_filename = self.cohort_directory / "cohort_info.json"
        with open(self.json_filename, 'w') as f:
            json.dump(self.cohort, f, indent = 4, sort_keys=True)

        

    def find_nwbs(self):
        # check if files from preliminary analysis exists for each session:
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])
                check_list = []
                nwb_file = str(self.find_file(session_folder, '.nwb'));

                self.cohort["mice"][mouse]["sessions"][session]["nwb_file"] = nwb_file  

                self.cohort["mice"][mouse]["sessions"][session]["phase"] = self.get_session_metadata(nwb_file)['phase']

    def init_raw_data(self):

        self.find_mice()

        self.check_raw_data()

        self.check_for_preliminary_analysis()

        # save as json
        self.json_filename = self.cohort_directory / "cohort_info.json"
        with open(self.json_filename, 'w') as f:
            json.dump(self.cohort, f, indent = 4, sort_keys=True)

        # save concise logs as json
        self.make_concise_training_logs()
        self.concise_json_filename = self.cohort_directory / "concise_cohort_info.json"
        with open(self.concise_json_filename, 'w') as f:
            json.dump(self.cohort_concise, f, indent = 4, sort_keys=True)

        if self.plot:
            self.graphical_cohort_info()

        print('Cohort info loaded')
    
    def get_session(self, ID, concise = False):
        """
        Takes a session ID and returns a dictionary containing the info from cohort info about that session.
        """
        # Iterate over each mouse and their sessions to find the matching session ID
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                if session == ID:
                    # Return the session information if the ID matches
                    if concise == False:
                        return self.cohort["mice"][mouse]["sessions"][session]
                    if concise == True:
                        return self.cohort_concise["complete_data"][mouse][session]
        # Return None or raise an error if the session ID is not found
        return None 

    def phases(self):
        """
        Makes a way of seeing the sessions that occured for each phase possible in a cohort.
        """
        phases = ["1", "2", "3", "3b", "3c", "4", "4b", "4c", "test", "5", "6", "7", "8", "9", "9b", "9c", "10"]
        phase_dict = {phase: {} for phase in phases}
        
        for mouse in self.cohort_concise["complete_data"]:
            for session in self.cohort_concise["complete_data"][mouse]:
                phase = self.cohort_concise["complete_data"][mouse][session]["Behaviour_phase"]
                session_path = self.cohort["mice"][mouse]["sessions"][session]["directory"]
                phase_dict[phase][session] = {"path": session_path,
                                              "total_trials": self.cohort_concise["complete_data"][mouse][session]["total_trials"],
                                              "video_length": self.cohort_concise["complete_data"][mouse][session]["video_length"],
                                              "mouse": mouse}
                
        return phase_dict

    def make_concise_training_logs(self):
        """
        concise form of logs for easy viewing of cohort.
        Split into complete and incomplete data
        For each session per mouse, include behaviour phase, total trials and length of video"""

        cohort_info = self.cohort
        self.cohort_concise = {}

        for mouse in cohort_info["mice"]:
            for session in cohort_info["mice"][mouse]["sessions"]:
                try:
                    behaviour_phase = cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["session_metadata"]["phase"]
                    total_trials = cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["session_metadata"]["total_trials"]
                    video_length = cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["video_length"]
                    if cohort_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                        if "complete_data" not in self.cohort_concise:
                            self.cohort_concise["complete_data"] = {}
                        if mouse not in self.cohort_concise["complete_data"]:
                            self.cohort_concise["complete_data"][mouse] = {}
                        self.cohort_concise["complete_data"][mouse][session] = {}
                        self.cohort_concise["complete_data"][mouse][session]["Behaviour_phase"] = behaviour_phase
                        self.cohort_concise["complete_data"][mouse][session]["total_trials"] = total_trials
                        self.cohort_concise["complete_data"][mouse][session]["video_length"] = video_length
                        self.cohort_concise["complete_data"][mouse][session]["mouse_id"] = mouse
                    else:
                        if "incomplete_data" not in self.cohort_concise:
                            self.cohort_concise["incomplete_data"] = {}
                        if mouse not in self.cohort_concise["incomplete_data"]:
                            self.cohort_concise["incomplete_data"][mouse] = {}
                        self.cohort_concise["incomplete_data"][mouse][session] = {}   
                        self.cohort_concise["incomplete_data"][mouse][session]["Behaviour_phase"] = behaviour_phase
                        self.cohort_concise["incomplete_data"][mouse][session]["total_trials"] = total_trials
                        self.cohort_concise["incomplete_data"][mouse][session]["video_length"] = video_length
                        self.cohort_concise["complete_data"][mouse][session]["mouse_id"] = mouse
                except:


                    ignore = ["240807_163610_wtjp254-4b",   # forgot to turn scales on
                            "240725_110604_wtjx300-6a",     # no trials
                            "240720_115745_wtjx300-6a",     # no trials
                            "240731_120318_wtjx300-6a",
                            "240720_114019_wtjx300-6a",
                            "240720_120336_wtjx300-6a",
                            "240720_120413_wtjx300-6a",
                            "240725_110604_wtjx300-6b",     # no trials
                            "240731_120318_wtjx300-6b",
                            "240720_114019_wtjx300-6b",
                            "240720_120336_wtjx300-6b",
                            "240720_120413_wtjx300-6b",
                            "240719_150822_wtjx261-2a",
                            "240716_141151_wtjx261-2a",
                            "240719_150822_wtjx307-6b"]
                    

                    if session in ignore:
                        continue
                    else:
                        print(f"Error processing {session}")
                    continue

    def graphical_cohort_info(self, show = False):
        # Assume self.cohort_concise["complete_data"] is already a suitable dictionary to convert to a DataFrame
        data = pd.DataFrame(self.cohort_concise["complete_data"])
        data.reset_index(inplace=True)
        data.rename(columns={"index": "SessionID"}, inplace=True)

        # Extract and process data
        data['SessionDate'] = pd.to_datetime(data['SessionID'].str[:6], format='%y%m%d').dt.date
        data_melted = data.melt(id_vars=["SessionID", "SessionDate"], var_name="Mouse", value_name="Details")
        data_melted.dropna(subset=["Details"], inplace=True)
        data_melted["BehaviourPhase"] = data_melted["Details"].apply(lambda x: x.get('Behaviour_phase'))
        data_melted["TotalTrials"] = data_melted["Details"].apply(lambda x: x.get('total_trials'))
        data_melted["VideoLength"] = data_melted["Details"].apply(lambda x: x.get('video_length'))

        # Summarize total trials
        total_trials = data_melted.groupby(['SessionDate', 'Mouse'])['TotalTrials'].sum().unstack()

        # Prepare unique behavior phases list
        # behavior_phases = data_melted.groupby(['SessionDate', 'Mouse'])['BehaviourPhase'].apply(lambda x: ', '.join(x.unique())).unstack()
        behavior_phases = data_melted.groupby(['SessionDate', 'Mouse'])['BehaviourPhase'].apply(lambda x: ', '.join(x)).unstack()
        # Summarize total sessions for annotations
        total_sessions = data_melted.groupby(['SessionDate', 'Mouse'])['SessionID'].nunique().unstack()

        # Plotting
        plt.figure(figsize=(18, 8))
        # Use total_trials for the heatmap values
        cbar_kws = {"label": "Total Num Trials"}  # Colorbar labeling
        ax = sns.heatmap(total_trials, cmap="viridis", linewidths=.5, cbar=True, cbar_kws=cbar_kws)
        ax.set_title(f'Total Trials and Behavior Phases by Mouse and Date (Complete data only)')

        # Annotations for total sessions and behavior phases
        for y in range(total_trials.shape[0]):
            for x in range(total_trials.shape[1]):
                session_count = total_sessions.iloc[y, x] if total_sessions.iloc[y, x] == total_sessions.iloc[y, x] else 0  # Check for NaN
                behavior_text = behavior_phases.iloc[y, x] if pd.notna(behavior_phases.iloc[y, x]) else ""
                # Annotate with total sessions count
                ax.text(x + 0.5, y + 0.5, f'Num Sess: {int(session_count)}', ha='center', va='center', color='black', fontsize='small', fontweight='bold', 
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
                # Annotate with behavior phases below the session count
                ax.text(x + 0.5, y + 0.8, f'Phases: {behavior_text}', ha='center', va='bottom', color='black', fontsize='small', fontweight='bold', wrap=True, 
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        filename = self.cohort_directory / "cohort_info.png"
        plt.savefig(filename, dpi=600)  # Adjust the dpi for resolution preferences

        if show:
            plt.show()
        # close figure:
        plt.close()

    def text_summary_cohort_info(self):
        # Assume self.cohort_concise["complete_data"] is already a suitable dictionary to convert to a DataFrame
        data = pd.DataFrame(self.cohort_concise["complete_data"])
        data.reset_index(inplace=True)
        data.rename(columns={"index": "SessionID"}, inplace=True)

        # Extract and process data
        data['SessionDate'] = pd.to_datetime(data['SessionID'].str[:6], format='%y%m%d').dt.date
        data_melted = data.melt(id_vars=["SessionID", "SessionDate"], var_name="Mouse", value_name="Details")
        data_melted.dropna(subset=["Details"], inplace=True)
        data_melted["BehaviourPhase"] = data_melted["Details"].apply(lambda x: x.get('Behaviour_phase'))
        data_melted["TotalTrials"] = data_melted["Details"].apply(lambda x: x.get('total_trials'))
        data_melted["VideoLength"] = data_melted["Details"].apply(lambda x: x.get('video_length'))

        # Group by session date
        grouped = data_melted.groupby('SessionDate')

        # Print summary
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

        # Optionally, you can save the summary to a text file
        filename = self.cohort_directory / "cohort_summary.txt"
        with open(filename, 'w') as file:
            file.write(summary_text)

    def find_mice(self):

        if not self.multi:
            # List of subdirectories in the test directory, only if index 13 is "_" and does not contain "OEAB_recording"
            self.session_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if len(folder.name) > 13 and folder.name[13] == "_" and folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
        else:
            self.multi_folders = [
                folder for folder in self.cohort_directory.glob('*')
                if folder.is_dir() and 'OEAB_recording' not in folder.name
            ]
            
            self.session_folders = [
                subfolder for folder in self.multi_folders for subfolder in folder.glob('*')
                if subfolder.is_dir() and len(subfolder.name) > 13 and subfolder.name[13] == "_" and 'OEAB_recording' not in subfolder.name
            ]

        cohort = {"Cohort name": self.cohort_directory.name, "mice": {}}
        for session_folder in self.session_folders:
            # get mouse ID:
            mouse_ID = session_folder.name[14:]
            # if mouse ID not in mice, add it:
            if mouse_ID not in cohort["mice"]:
                cohort["mice"][mouse_ID] = {"sessions": {}}
            # add session folder to mice:
            session_ID = session_folder.name
            cohort["mice"][mouse_ID]["sessions"][f"{session_ID}"] = {"directory": (str(session_folder)),
                                                                     "mouse_id": mouse_ID,
                                                                     "session_id": session_ID}

        self.cohort = cohort





    def check_raw_data(self):
        # check if raw data exists for each session:
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])
                raw_data = {}
                check_list = []
                raw_data["raw_video"] = str(self.find_file(session_folder, '.avi'));
                raw_data["behaviour_data"] = str(self.find_file(session_folder, 'behaviour_data')); check_list.append(raw_data["behaviour_data"])
                raw_data["tracker_data"] = str(self.find_file(session_folder, 'Tracker_data')); check_list.append(raw_data["tracker_data"])
                raw_data["arduino_DAQ"] = str(self.find_file(session_folder, 'ArduinoDAQ')); check_list.append(raw_data["arduino_DAQ"])
                raw_data["OEAB"] = str(self.find_OEAB_dir(session_folder, mouse)); check_list.append(raw_data["OEAB"])

                bmp_file_check = True if self.find_file(session_folder, 'temp') != None else False
                video_check = "None" if raw_data["raw_video"] == "None" and bmp_file_check == False else True
                check_list.append(video_check)

                # load behaviour data:
                if raw_data["behaviour_data"] != "None":
                    try:
                        with open(raw_data["behaviour_data"]) as f:
                            data = json.load(f)
                        if len(data["Scales data"]) > 0:
                            raw_data["scales_data"] = True; check_list.append(raw_data["scales_data"])
                        else:
                            raw_data["scales_data"] = False; check_list.append("None")
                    except:
                        raw_data["scales_data"] = False; check_list.append("None")
                else:
                    raw_data["scales_data"] = False; check_list.append("None")

                if "None" not in check_list:
                    raw_data["is_all_raw_data_present?"] = True
                else:
                    raw_data["is_all_raw_data_present?"] = False
                
                # make list of missing files:
                missing_files = []
                missing_files.append("raw_video") if raw_data["raw_video"] == "None" else None
                missing_files.append("behaviour_data") if raw_data["behaviour_data"] == "None" else None
                missing_files.append("tracker_data") if raw_data["tracker_data"] == "None" else None
                missing_files.append("arduino_DAQ") if raw_data["arduino_DAQ"] == "None" else None
                missing_files.append("OEAB") if raw_data["OEAB"] == "None" else None

                raw_data["missing_files"] = missing_files

                # get OEAB file info:
                OEAB_contents = self.get_OEAB_file_info(raw_data["OEAB"])
                raw_data["OEAB_contents"] = OEAB_contents

                # get length of video:
                raw_data["video_length"] = self.get_video_length(raw_data["raw_video"])

                # get session metadata:
                raw_data["session_metadata"] = self.get_session_metadata(raw_data["behaviour_data"])

                # add raw data to cohort:
                self.cohort["mice"][mouse]["sessions"][session]["raw_data"] = raw_data

    def get_video_length(self, video_file):
        if video_file != "None":
            video = cv.VideoCapture(video_file)
            fps = video.get(cv.CAP_PROP_FPS)
            frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            video.release()
            # Calculate the video duration in seconds and then convert to minutes and seconds
            duration_seconds = frame_count / fps if fps else 0
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            
            return f"{minutes}m{seconds}s"
            # return f"{video_file}"
        else:
            return None
    
    def get_OEAB_file_info(self, OEAB_file):
        OEAB_contents = {}
        # count total files, including those in subdirectories:
        if OEAB_file != "None":
            OEAB_nodes = list(Path(OEAB_file).glob('*'))
            OEAB_contents["nodes"] = [str(node) for node in OEAB_nodes]
        else:
            OEAB_nodes = None
            OEAB_contents["nodes"] = None
        

        if OEAB_nodes != None:
            if OEAB_nodes[0].name == "Record Node 113" or OEAB_nodes[0].name == "Record Node 101":
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
    
    def get_session_metadata(self, behaviour_data_file):
        """
        Gets behaviour phase and total trials from behaviour data file.
        """
        session_metadata = {}

        try:
            if behaviour_data_file != "None":
                if Path(behaviour_data_file).suffix == '.json':
                    with open(behaviour_data_file) as f:
                        data = json.load(f)
                    session_metadata["phase"] = data["Behaviour phase"]
                    session_metadata["total_trials"] = data["Total trials"]
                if Path(behaviour_data_file).suffix == '.nwb':
                    with NWBHDF5IO(str(behaviour_data_file), 'r') as io:
                        nwbfile = io.read()
                        nwb_metadata = nwbfile.experiment_description
                        phase = nwb_metadata.split(";")[0].split(":")[1]
                    session_metadata['phase'] = phase
                    session_metadata["total_trials"] = None
            else:
                session_metadata["phase"] = None
                session_metadata["total_trials"] = None
        except:
            # print traceback
            traceback.print_exc()
            return None





        return session_metadata


    def check_for_preliminary_analysis(self):
        # check if files from preliminary analysis exists for each session:
        for mouse in self.cohort["mice"]:
            for session in self.cohort["mice"][mouse]["sessions"]:
                session_folder = Path(self.cohort["mice"][mouse]["sessions"][session]["directory"])
                processed_data = {}
                check_list = []
                processed_data["processed_DAQ_data"] = str(self.find_file(session_folder, 'processed_DAQ_data')); check_list.append(processed_data["processed_DAQ_data"])
                processed_data["sendkey_logs"] = str(self.find_file(session_folder, 'sendkey_logs')); check_list.append(processed_data["sendkey_logs"])
                processed_data["video_frametimes"] = str(self.find_file(session_folder, 'video_frame_times')); check_list.append(processed_data["video_frametimes"])
                processed_data["sendkey_metadata"] = str(self.find_file(session_folder, 'behaviour_data')); check_list.append(processed_data["sendkey_metadata"])
                processed_data["NWB_file"] = str(self.find_file(session_folder, '.nwb')); #check_list.append(processed_data["NWB_file"])
                processed_data["DLC"] = self.find_DLC_files(session_folder);
                if "None" not in check_list:
                    processed_data["preliminary_analysis_done?"] = True
                else:
                    processed_data["preliminary_analysis_done?"] = False

                self.cohort["mice"][mouse]["sessions"][session]["processed_data"] = processed_data

    def find_DLC_files(self, session_folder):
        DLC_files = {}
        DLC_files["labeled_video"] = str(self.find_file(session_folder, 'labeled'))
        DLC_files["coords_csv"] = str(self.find_file(session_folder, '800000.csv'))
        return DLC_files
    
    # def sort_sessions(self):
    #     # sort sessions by date:
    #     for mouse in self.cohort["mice"]:
    #         # sort sessions by date:
    #         for session in self.cohort["mice"][mouse]["sessions"]:
    #             session_ID = session
    #             session_date = session_ID[:13]
    #             self.cohort["mice"][mouse]["sessions"][session]["date"] = session_date


    def find_file(self, directory, tag):
        for file in directory.glob('*'):
                if tag in file.name:
                    return file
        return None

    def find_OEAB_dir(self, directory, mouse):
        """
        Finds the OEAB data by looking for the folder that contains 'OEAB_recording'.
        """
        if not self.multi:
            for file in directory.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            return None
        else:
            parent = directory.parent
            for file in parent.glob('*'):
                if file.is_dir() and 'OEAB_recording' in file.name:
                    return file
            return None

def main():

    test_dir = Path(r"/cephfs2/srogers/Behaviour code/2407_July_WT_cohort/Portable_data")

    cohort = Cohort_folder(test_dir, multi = True, plot = True, portable_data = True)

    # cohort.graphical_cohort_info()
    # print(cohort.get_session("240311_183300"))


if __name__ == "__main__":
    main()





