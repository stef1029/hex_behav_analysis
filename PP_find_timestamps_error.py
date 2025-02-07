from Session_nwb import Session
from Cohort_folder import Cohort_folder
from pathlib import Path
import traceback

# Go through every session and print(sess.timestamps[0], sess.timestamps[-1]):

cohort_directory = r"/cephfs2/srogers/March_training"

cohort = Cohort_folder(cohort_directory, multi = True)
cohort_info = cohort.cohort

sessions = []

log_file = 'PP-find_timestamps_error.txt'

for mouse in cohort_info['mice']:
    for session in cohort_info['mice'][mouse]['sessions']:
        sessions.append(session)

session_count = 0
for session in sessions:
    session_count += 1
    print(f"Session {session_count}/{len(sessions)}: {session}")
    with open(log_file, 'a') as f:
        f.write(f"Session {session_count}/{len(sessions)}: {session}\n")
    try:
        sess = Session(cohort.get_session(session))
        if len(sess.trials) > 0:
            print(sess.trials[0]['cue_start'], sess.trials[-1]['cue_start'])
            with open(log_file, 'a') as f:
                f.write(f"{session}: {sess.trials[0]['cue_start']} {sess.trials[-1]['cue_start']}\n")

    except Exception as e:
        print(f"Error in session {session}: {e}")
        # print traceback:
        print(traceback.format_exc())
        with open(log_file, 'a') as f:
            f.write(f"Error in session {session}: {e}\n")
            f.write(traceback.format_exc())


        continue
    