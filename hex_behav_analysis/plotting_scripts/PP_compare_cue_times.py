"""
This script is for plotting the results of the cue times experiment.
Each of the sessions in the experiment have different cue times, and the goal is to compare the performance of the mice in each session.
This should also be done based on cue presentation angle.
Each session will also have a learning/ performance variation across the hour, and I would like to plot this too,
however may have to just do this based on overall performance because of the number of trials, 
since currently each hour only has about 50-60 trials per port.

So, this should produce first a plot of performance over time for each cue time, layer on each other.
Then, each session will be line-plotted, left and right turns, all on the same plot. Radial plots can also be made for visualisation.
I will do linear regression on these lines and then prepare a third plot of the slopes of these lines.
"""


#%%

from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
import time

from utils.Cohort_folder import Cohort_folder
from utils.Session import Session

cohort_directory = Path(r"/cephfs2/srogers/March_training")

cohort = Cohort_folder(cohort_directory, multi = True)

phases = cohort.phases()

# collect all phase 9c sessions:
sessions = []
phase = "9c"
for session in phases[phase]:
    # date = session[:6]
    sessions.append(Path(phases[phase][session]["path"]))

#%%

# load in sessions:
Sessions = []
num_sessions = len(sessions)
start_time = time.perf_counter()
for session in sessions:
    print(f"Loading session {session}... ({sessions.index(session)+1}/{num_sessions})")
    Sessions.append(Session(session))
    # print time so far in minutes and seconds, rounded:
    print(f"Time so far: {round((time.perf_counter()-start_time)//60)} minutes, {round((time.perf_counter()-start_time)%60)} seconds")

print(f"Total time: {round((time.perf_counter()-start_time)//60)} minutes, {round((time.perf_counter()-start_time)%60)} seconds")

#%%