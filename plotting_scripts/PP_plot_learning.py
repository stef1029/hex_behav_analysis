import matplotlib.pyplot as plt
import numpy as np
from Session import Session
from Cohort_folder import Cohort_folder
from pathlib import Path



def plot_learning(session):
    

    trials = session.trials

    bin_size = 5  # minutes

    # Assuming 'session.timestamps' is a dictionary where trials' start and possibly end are mapped to timestamps in seconds
    first_trial_time = session.timestamps[trials[0]["start"]] / 60  # convert to minutes
    last_trial_time = session.timestamps[trials[-1]["start"]] / 60  # convert to minutes

    # We calculate the total duration to include all trial times, not rounding it at this point
    total_session_time = last_trial_time - first_trial_time

    # Initialize dict to hold the sum of trials for each bin
    bins = {i: [] for i in np.arange(first_trial_time, first_trial_time + total_session_time, bin_size)}

    for trial in trials:
        start_time = session.timestamps[trial['start']] / 60  # convert to minutes
        
        bin_index = int((start_time - first_trial_time) // bin_size) * bin_size + first_trial_time
        bins[bin_index].append(1)

    # Now adjust the counts in bins to be an average per actual time in each bin, dealing with the last bin separately
    x = []
    y = []
    for bin_start, trials in bins.items():
        bin_duration = bin_size if bin_start + bin_size <= total_session_time + first_trial_time else total_session_time + first_trial_time - bin_start
        x.append(bin_start)
        y.append(len(trials) / (bin_duration / bin_size))  # Normalize count by the actual duration of the bin in units of 'bin_size'

    plt.plot(x, y)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average Number of Trials')
    plt.title('Average Trials per Bin (Normalised)')
    plt.show()

if __name__ == "__main__":

    cohort_directory = r"/cephfs2/srogers/December_training_data"

    cohort = Cohort_folder(cohort_directory)

    session_directory = r"/cephfs2/srogers/240207_Dans_data/240206_121205_WTJP239-4a"

    info = cohort.cohort

    # session_directories = {}
    # for mouse in info['mice']:
    #     session_directories[mouse] = {}
    #     session_num = 0
    #     for session in info['mice'][mouse]['sessions']:
    #         if info['mice'][mouse]['sessions'][session]['processed_data']['preliminary_analysis_done?'] == True:
    #             directory = Path(info['mice'][mouse]['sessions'][session]['directory'])
    #             phase = info['mice'][mouse]['sessions'][session]['raw_data']['session_metadata']['phase']
    #             if directory.exists() and phase != 'test':
    #                 session_directories[mouse][session_num] = ({'dir': str(directory), 'phase': phase})
    #                 session_num += 1

    # for mouse in session_directories:
    #     for session in session_directories[mouse]:
    #         print(session, session_directories[mouse][session])
        # plot_learning(session)
            
    session_directories = {}
    for mouse in info['mice']:
        sessions = []
        for session in info['mice'][mouse]['sessions']:
            if (info['mice'][mouse]['sessions'][session]['processed_data']['DLC']['coords_csv'] != 'None'):
                directory = Path(info['mice'][mouse]['sessions'][session]['directory'])
                phase = info['mice'][mouse]['sessions'][session]['raw_data']['session_metadata']['phase']
                if directory.exists() and phase != 'test':
                    sessions.append((str(directory), phase))
                    
        # Sort sessions by directory (which includes datetime in its name)
        sorted_sessions = sorted(sessions, key=lambda x: x[0])
        
        # Populate session_directories after sorting
        session_directories[mouse] = {i: {'dir': dir_phase[0], 'phase': dir_phase[1]} for i, dir_phase in enumerate(sorted_sessions)}

    # Print the sorted sessions
    # session_list = session_directories['WTJP239-4a']
    for mouse in session_directories:
        for session in session_directories[mouse]:
            print(session, session_directories[mouse][session]['dir'])
            # plot_learning(Session(session_list[session]['dir'])) 
            d = session_directories[mouse][session]['dir']
            sess = Session(d)
            print(sess.timestamps[0], sess.timestamps[-1])
        
    # session_directory = r"/cephfs2/srogers/240207_Dans_data/240207_121732_WTJP239-4a"

    # session = Session(session_directory)

    # trials = session.trials

    # print(trials[0])

    # plot_learning(session)