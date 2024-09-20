
def plot_performance_by_angle(cohort, sessions, bin_mode='manual', num_bins=12, trials_per_bin=10, plot_mode='bar'):
    import matplotlib.pyplot as plt
    import numpy as np

    # Load trial list
    trials = []
    for session in sessions:
        session = Session(cohort.get_session(session))
        trials += session.trials

    # Determine number of bins
    n = len(trials)
    if bin_mode == 'manual':
        pass
    elif bin_mode == 'rice':
        num_bins = int(2 * n ** (1/3))
    elif bin_mode == 'tpb':
        num_bins = int(n / trials_per_bin)
    else:
        raise ValueError('bin_mode must be "manual", "rice", or "tpb"')

    # Initialize bins for left and right turns
    bin_size = round(180 / num_bins)
    left_bins = {i: [] for i in range(0, 180, bin_size)}
    right_bins = {i: [] for i in range(0, 180, bin_size)}

    # Bin trials based on turn direction and angle
    for trial in trials:
        if trial["turn_data"] is not None:
            angle = trial["turn_data"]["cue_presentation_angle"]
            correct = int(trial["correct_port"]) == int(trial["next_sensor_ID"])

            if angle < 0:  # Left turn
                bin_index = abs(angle) // bin_size * bin_size
                left_bins[bin_index].append(correct)
            elif angle > 0:  # Right turn
                bin_index = angle // bin_size * bin_size
                right_bins[bin_index].append(correct)

    # Calculate performance for each bin
    def calc_performance(bins):
        return [sum(bins[key]) / len(bins[key]) if bins[key] else 0 for key in sorted(bins)]

    left_performance = calc_performance(left_bins)
    right_performance = calc_performance(right_bins)
    bin_titles = [f"{int(key) + (bin_size / 2)}" for key in sorted(left_bins)]

    # Plotting function
    def plot_performance(bin_titles, performance, title, color_map='viridis'):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        colors = plt.cm.get_cmap(color_map, len(bin_titles))(range(len(bin_titles)))
        plt.bar(bin_titles, performance, color=colors, edgecolor='black', width=1)
        plt.xlabel('Turn Angle (degrees)', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(bin_titles, rotation=45)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Generate plots
    if plot_mode == 'bar':
        plot_performance(bin_titles, left_performance, 'Left Turn Performance')
        plot_performance(bin_titles, right_performance, 'Right Turn Performance')
