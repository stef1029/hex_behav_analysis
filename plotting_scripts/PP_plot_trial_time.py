from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

cohort_directory = Path(r"/cephfs2/srogers/240207_Dans_data")

cohort = Cohort_folder(cohort_directory)

phases = cohort.phases()

# print(phases["9"])

sessions = []
for session in phases["9"]:
    date = session[:6]
    if date == "240213":
        sessions.append(Path(phases["9"][session]["path"]))

# sessions = [sessions[0]]

trials = []
session_objects = []
for i, session in enumerate(sessions):
    session = Session(session)
    session_objects.append(session)
    # trials.append([(i, trial) for trial in session.trials if trial["turn_data"] != None])
    for trial in session.trials:
        if trial["turn_data"] != None:
            trials.append((i, trial))

# bin the trials into 30 degree bins, ranging from 180 to -180
n = len(trials)
num_bins = n / 40

# bin num by rice rule:
num_bins = 2 * n ** (1/3)
bin_size = round(360 / num_bins)
bin_size = 30

bins = {i: [] for i in range(-180, 180, bin_size)}

for tuple in trials:
    (session_index, trial) = tuple
    if trial["turn_data"] != None:
        angle = trial["turn_data"]["cue_presentation_angle"]
        for bin in bins:
            if angle < bin + bin_size and angle >= bin:
                trial_start = session_objects[session_index].timestamps[trial["start"]]
                trial_end = session_objects[session_index].timestamps[trial["next_sensor_time"]]

                trial_time = trial_end - trial_start
                success = True if int(trial["correct_port"]) == int(trial["next_sensor_ID"]) else False

                if success:
                    bins[bin].append(trial_time)

bin_titles = []
performance = []
trial_counts = []
for key in bins:
    if len(bins[key]) > 0:
        average = sum(bins[key]) / len(bins[key])
        median = np.median(bins[key])
        length = len(bins[key])
    else:
        average = 0  # Default to 0 (or another placeholder value) when there are no trials in the bin
        median = 0
        length = 0
    # bins[key] = median
    performance.append(average)
    bin_titles.append(f"{int(key) + (bin_size/2)}")
    trial_counts.append(length)

scaled_sizes = [1 + (count * 0.2) for count in trial_counts]

# Improved plotting
plt.figure(figsize=(10, 6))  # Set the figure size for better readability

# Use a more visually appealing style. You can choose any from plt.style.available
plt.style.use('ggplot') 

# Generating colors - a simple way to add different colors to the bars for visual differentiation
colors = plt.cm.viridis(np.linspace(0, 1, len(bin_titles)))

# Plotting with enhancements
plt.bar(bin_titles, performance, color=colors, edgecolor='black', width = 1)

# Adding labels and title with improved font sizes
plt.xlabel('Left turns -- Turn angle -- Right Turns', fontsize=14)
plt.ylabel('Trial time', fontsize=14)
plt.title('Time to get reward by turn angle', fontsize=16)

# Adding x-ticks to ensure they are clearly readable; rotating them for better layout
plt.xticks(bin_titles, rotation=45)

# Adding gridlines for better readability; configuring them to be lighter and dashed
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# Fine-tuning: adjusting the margins and layout for better spacing and fit
plt.gcf().subplots_adjust(bottom=0.15)  # Adjust bottom to accommodate x-labels
plt.tight_layout()

# Save plot with specified DPI for higher resolution
plt.savefig("/cephfs2/srogers/test_output/times_by_angle_bar.png", dpi=300)


# Radial Plot:

# Preparation of the data remains the same
angles_deg = np.array(bin_titles, dtype=np.float64)  # Original angles, from -180 to 180
performance_data = np.array(performance)  # Assuming performance data is ready

# Adjust angles for plotting and convert to radians
adjusted_angles_deg = angles_deg % 360  # Adjust for radial plot
angles_rad = np.radians(adjusted_angles_deg)  # Convert to radians

bins_std = {}
for key, times in bins.items():
    bins_std[key] = np.std(times)  # Using standard deviation as a measure

iqr_values = {}  # Placeholder for IQR values for each bin.
for key, times in bins.items():
    Q1 = np.percentile(times, 25)
    Q3 = np.percentile(times, 75)
    iqr_values[key] = Q3 - Q1

# Convert iqr_values to a list or array in the same order as performance_data
iqr = [iqr_values[key] for key in sorted(bins)]  # Assuming bin sorting matches
iqr = np.append(iqr, iqr[0])  # Closing the plot

# Convert bins_std to a list or array in the same order as performance_data
std_dev = [bins_std[key] for key in sorted(bins)]

# Append the start to the end to close the plot
angles_rad = np.append(angles_rad, angles_rad[0])
performance_data = np.append(performance_data, performance_data[0])
std_dev = np.append(std_dev, std_dev[0])



# Create radial plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Polar plot with adjustments
ax.plot(angles_rad, performance_data, marker='o')  # Add markers for data points

# for angle, performance, size in zip(angles_rad, performance_data, scaled_sizes):
#     ax.plot(angle, performance, 'o', color='red', markersize=size)

# ax.fill(angles_rad, performance_data, alpha=0.25)  # Fill for visual emphasis

# # Shaded area for variance
# lower = performance_data - std_dev
# upper = performance_data + std_dev
# ax.fill_between(angles_rad, lower, upper, alpha=0.2, label='Std. Dev.')

# Shaded area for IQR
lower = performance_data - iqr / 2  # Assuming you want to center the IQR around the median
upper = performance_data + iqr / 2
ax.fill_between(angles_rad, lower, upper, alpha=0.2, label='IQR')

# Adjusting tick labels to reflect left (-) and right (+) turns
tick_locs = np.radians(np.arange(-180, 181, 30)) % (2 * np.pi)  # Tick locations, adjusted for wrapping
tick_labels = [f"{int(deg)}" for deg in np.arange(-180, 181, 30)]  # Custom labels from -180 to 180

ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels)

# Custom plot adjustments
ax.set_theta_zero_location('N')  # Zero degrees at the top for forward direction
ax.set_theta_direction(1)  # AntiClockwise direction

# Add title
ax.set_title('Mean Trial Time by Turn Angle (s) (Successful trials)', va='bottom', fontsize=16)

# add text in bottom right:
text = f"Trials: {n} - Mice: {len(sessions)}"
ax.text(0, 0, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

ax.legend(loc='best')

# Optionally, save the plot with a specific filename
plt.savefig("/cephfs2/srogers/test_output/success_times_by_angle_radial.png", dpi=300)