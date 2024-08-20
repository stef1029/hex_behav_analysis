from Session import Session
from matplotlib import pyplot as plt
import json
import numpy as np
from pathlib import Path
import importlib

# Session = importlib.import_module('Session')

def print_dict_structure(d, indent=0):
    print("Keys:")
    for key, value in d.items():
        print('    ' * indent + str(key), end=': ')
        if isinstance(value, dict):
            # It's another dictionary, so repeat the process on this value
            print()
            print_dict_structure(value, indent+1)
        elif isinstance(value, list):
            # It's a list, note its size but don't print its contents
            print('List of {} items'.format(len(value)))
        else:
            # It's neither a list nor a dictionary, so just note it's a value
            print('Value')

if __name__ == "__main__":
    
    session_directory = Path(r"/cephfs2/srogers/December_training_data/231220_115610_wtjx285-2d")

    session = Session(session_directory)

    trials = session.trials

    # print_dict_structure(trials[0])
    print(trials[0]["DLC_data"])

    DLC_performance_nose = []
    for trial in trials:
        average_confidence = np.mean(trial["DLC_data"]["left_ear"]["likelihood"][:15])
        print(average_confidence)
        DLC_performance_nose.append(average_confidence)

    

