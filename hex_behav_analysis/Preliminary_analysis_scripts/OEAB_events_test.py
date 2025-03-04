import os
import json
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import open_ephys.analysis


class process_ADC_Recordings:
    def __init__(self, dirname, rig=None):
        self.dirname = dirname
        self.rig = rig or 1

        self.extract_ADC_data()


    def extract_ADC_data(self):
        self.recording = open_ephys.analysis.Session(self.dirname).recordnodes[0].recordings[0].continuous[0]
        self.events = open_ephys.analysis.Session(self.dirname).recordnodes[0].recordings[0].events
        self.timestamps = self.recording.timestamps
        self.metadata = self.recording.metadata


        # events is a pandas dataframe, print 5 lines of it where the heading 'line' is 1:
        print(self.events[self.events['line'] == 1].head(5))




if __name__ == "__main__":
    # test = process_ADC_Recordings(r"E:\Test_output\240906_001430\240906_001430_OEAB_recording")
    test = process_ADC_Recordings(r"E:\Test_output\240906_024544\240906_024544_OEAB_recording")
    