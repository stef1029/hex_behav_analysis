import numpy as np
import matplotlib.pyplot as plt   

def plot_test(sessions):
    """
    This function takes a list of sessions where mice were performing the test phase. 
    In this phase mice are first presented with 5 trials with led at port 1 and then 10 at port 3.
    For each mouse, I wil create an array of port numbers touched, 1-6, and then 
    plot these over each other as the final plot, if it's not too messy. 
    Then to show average performance, I can do the two performance plots for port 1 and 3 per trial.
    """
    
    mice = {}
    min_length = float('inf')
    for session in sessions:
        mouse_id = session.session_dict['mouse_id']
        trials = session.trials
        ports_touched = [int(trial['next_sensor']['sensor_touched']) for trial in trials]

        if mouse_id in mice:
            mice[mouse_id].append(np.array(ports_touched))
        else:
            mice[mouse_id] = [np.array(ports_touched)]

        min_length = min(min_length, len(ports_touched))

    fig, ax = plt.subplots()
    for mouse, data in mice.items():
        truncated_data = [arr[:min_length] for arr in data]
        concatenated_data = np.vstack(truncated_data)
        ax.plot(np.mean(concatenated_data, axis=0), label=mouse)

    # Highlight the region after trial 4
    ax.axvspan(0, 4.45, color='red', alpha=0.1, label='Cue at port 1')
    ax.axvspan(4.5, min_length, color='blue', alpha=0.1, label='Cue at port 3')

    ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.2,1))
    ax.set_title('Port touched per trial')
    # set x axis label:
    ax.set_xlabel('Trial number')
    # set y axis label:
    ax.set_ylabel('Port touched')
    plt.show()

    # Performance analysis
    performance_1, performance_3 = [], []
    for mouse, data in mice.items():
        truncated_data = [arr[:min_length] for arr in data]
        concatenated_data = np.vstack(truncated_data)
        performance_1.append(np.sum(concatenated_data == 1, axis=0))
        performance_3.append(np.sum(concatenated_data == 3, axis=0))

    total_performance_1 = np.sum(np.array(performance_1), axis=0) / len(mice)
    total_performance_3 = np.sum(np.array(performance_3), axis=0) / len(mice)

    fig, ax = plt.subplots()
    ax.plot(total_performance_1, label='Port 1 performance')
    ax.plot(total_performance_3, label='Port 3 performance')
    ax.set_ylim(0, 1)

    # Highlight the region after trial 4 in the performance plot as well
    ax.axvspan(0, 4.45, color='red', alpha=0.1, label='Cue at port 1')
    ax.axvspan(4.5, min_length, color='blue', alpha=0.1, label='Cue at port 3')

    ax.legend(loc='upper right', fontsize='small')

    ax.set_title('Performance per trial')

    ax.set_xlabel('Trial number')
    ax.set_ylabel('Likelihood of port touch')
    plt.show()

