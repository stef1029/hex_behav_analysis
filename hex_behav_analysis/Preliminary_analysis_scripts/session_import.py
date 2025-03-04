import csv
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import open_ephys.analysis


class Trial:
    def __init__(self, trial, logs_data, datetime):
        self.trial = trial
        self.datetime = datetime
        self.logs_data = logs_data

        self.process_logs_data()



    def process_logs_data(self):

        self.computer_start_time = self.logs_data[0][1]
        self.computer_end_time = self.logs_data[-1][1]
        
        self.trial_duration_computer = self.computer_end_time - self.computer_start_time
        # self.trial_duration_arduino = self.arduino_end_time - self.arduino_start_time

        self.trial_success = True if self.logs_data[-1][-1] == "success" else False

        try:
            self.correct_port = self.logs_data[1][3]
        except:
            self.correct_port = "None"
            
        self.error_number = 0
        self.errors = []
        if self.trial_success == True:
            for i, item in enumerate(self.logs_data[2:]):
                if item[-1] == "failure":
                    self.error_number += 1
                    error_time_computer = item[1] - self.computer_start_time
                    # error_time_arduino = item[4] - self.arduino_start_time
                    error_port = item[3]
                    self.errors.append([self.error_number, error_port, error_time_computer])
        

    

    def __repr__(self):
        return f"""Trial: {str(self.trial)}, Success: {str(self.trial_success)}, Port: {self.correct_port}    
            \rComputer duration: {str(self.trial_duration_computer)}
            \rErrors: {str(self.error_number)}, Error details: {str(self.errors)}"""






class Session:
    """ The Session class contains a list of trials which occured in the session and the metadata for that session which is consistent for all trials."""

    def __init__(self, logs_path):
        self.logs_path = logs_path
        # have the logs ingestion functions within Session, and just give it the file to ingest and save into itself
        # then it'll populate itself with trials and keep the metadata at this higher level.
        self.ingest_json_file(self.logs_path)


    def ingest_json_file(self, json_file_path):
        """ This function ingests a json file and populates the Session object with the data from the file.
        It also creates a list of trials from the data in the session."""
        # load json file to dictionary:
        with open(json_file_path, 'r') as f:
            self.json_data = json.load(f)

        self.data = self.json_data.pop("Logs")
        # print(self.json_data)

        # make a list for lists of data, split by ;
        data_split = []
        for i in range(len(self.data)):
            data_split.append(self.data[i].split(";"))

        for i, value in enumerate(data_split):

            if len(value) == 2:
                value[0] = str(value[0])
                value[1] = float(value[1])

            if len(value) > 4:
                value[0] = str(value[0])
                value[1] = float(value[1])
                value[2] = str(value[2])
                value[3] = int(value[3]) if value[3] != 'F' else str(value[3])
                # try: 
                #     value[4] = int(value[4]); 
                # except ValueError: 
                #     value[4] = 0
                if value[-1] == 'T':
                    value[-1] = "success"
                elif value[-1] == 'F':
                    value[-1] = "failure"
                else:
                    value[-1] = "reply"

        self.data_split = data_split
        # print(self.data_split)

        # self.mouse_id = self.json_data['Mouse ID']
        # self.number_of_trials_set = self.json_data['Number of trials']
        # self.behaviour_phase = self.json_data['Behaviour phase']
        # self.scales_tested = self.json_data['Scales tested?']
        # self.mouse_weight = self.json_data['Mouse weight']
        # self.port = self.json_data['Port']
        # self.datetime = self.json_data['Date and time']
        # self.readable_datetime = f"{self.datetime[7:9]}:{self.datetime[9:11]}:{self.datetime[11:13]} {self.datetime[4:6]}/{self.datetime[2:4]}/{self.datetime[0:2]}"
        # self.total_trials = self.json_data['Total trials']

        # if "Mouse weight threshold" in self.json_data.keys():
        #     self.mouse_weight_threshold = self.json_data['Mouse weight threshold']
        # if "Platform pause time" in self.json_data.keys():
        #     self.platform_pause_time = self.json_data['Platform pause time']
        # if "Initial series length" in self.json_data.keys():
        #     self.initial_series_length = self.json_data['Initial series length']
        # if "Test length" in self.json_data.keys():
        #     self.test_length = self.json_data['Test length']
        # if "End time" in self.json_data.keys():
        #     self.session_end_time = self.json_data['End time']

        self.rig_id = self.json_data.get('Rig')
        self.mouse_id = self.json_data.get('Mouse ID')
        self.number_of_trials_set = self.json_data.get('Number of trials')
        self.behaviour_phase = self.json_data.get('Behaviour phase')
        self.scales_tested = self.json_data.get('Scales tested?')
        self.mouse_weight = self.json_data.get('Mouse weight')
        self.port = self.json_data.get('Port')
        self.datetime = self.json_data.get('Date and time')
        self.readable_datetime = f"{self.datetime[7:9]}:{self.datetime[9:11]}:{self.datetime[11:13]} {self.datetime[4:6]}/{self.datetime[2:4]}/{self.datetime[0:2]}" if self.datetime else None
        self.total_trials = self.json_data.get('Total trials')

        self.mouse_weight_threshold = self.json_data.get('Mouse weight threshold')
        self.platform_pause_time = self.json_data.get('Platform pause time')
        self.initial_series_length = self.json_data.get('Initial series length')
        self.test_length = self.json_data.get('Test length')
        self.session_end_time = self.json_data.get('End time')
        self.scales_data = self.json_data.get('Scales data')

        self.cue_duration = self.json_data.get('Cue duration')
        self.wait_duration = self.json_data.get('Wait duration')
        self.video_fps = self.json_data.get('FPS')
        self.catch_trial_type = self.json_data.get('Catch_trial_type')
        self.catch_brightness = self.json_data.get('Catch_brightness')
        self.catch_wait_time = self.json_data.get('Catch_wait')

        # if "Scales data" in self.json_data.keys():
        #     self.scales_data = self.json_data['Scales data']

        self.create_trials()
        


    def create_trials(self):
        """ This function creates a list of trials from the data in the session.
        It does this by finding the start and end of each trial and then creating a Trial object for each trial."""
        start = 0
        trial_number = 0
        self.trials = []
        for i, item in enumerate(self.data_split):
            if i == 0: continue

            try: 
                if len(self.data_split[i+1]) == 2:
                    trial_data = self.data_split[start:i+1]
                    self.trials.append(Trial(trial_number, trial_data, self.datetime))
                    start = i + 1
                    trial_number += 1
            except IndexError:
                trial_data = self.data_split[start:i+1]
                self.trials.append(Trial(trial_number, trial_data, self.datetime))
                start = i + 1
                trial_number += 1
        
    def ingest_ephys_files(self, ephys_path):
        # Split ephys into spike sorting pipeline and TTL tracking system.
        pass


    def dataframe(self):
        data = []
        for trial in self.trials:
            data.append([trial.trial, 
                         trial.trial_success, 
                         trial.correct_port, 
                         trial.trial_duration_computer, 
                         trial.error_number])
        self.df = pd.DataFrame(data, columns=["Trial", 
                                                "Success", 
                                                "Port", 
                                                "Computer duration", 
                                                "Errors"])
        return self.df
    
    def metadata(self):
        print(f"{self.readable_datetime} Session metadata:")
        for key in self.json_data:
            print(f"{key}: {self.json_data[key]}")

    
    def __repr__(self):
        self.metadata()

