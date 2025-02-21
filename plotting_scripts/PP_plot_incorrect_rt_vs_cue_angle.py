import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import math
from statistics import mean, stdev

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

def plot_incorrect_rt_vs_cue_angle_abs(
    sessions_list,
    output_path=None,
    title="Incorrect Trials: Reaction Time vs. ABS Cue Angle",
    front_limit=60,
    angle_max=180,
    num_bins=12,
    error_bars='sem',  # 'sem' or 'sd'
    color=(0.93, 0, 0.55)  # e.g. "visual_trials" style
):
    """
    Plots mean reaction time (only for incorrect trials) vs. the absolute value 
    of the cue presentation angle, while also showing a line plot of the 
    trial counts per bin on a secondary y-axis.

    - The primary y-axis shows Mean Reaction Time (s).
    - The secondary y-axis shows Number of Trials (n).
    - Both y-axes start at 0, and we manually set the y-max to ensure 
      all data and error bars are visible.

    Parameters
    ----------
    sessions_list : list
        List of session objects.
    output_path : str or Path, optional
        Where to save the figure. If None, won't save.
    title : str
        Title for the plot.
    front_limit : float
        Only include trials if port_touched_angle is within ±front_limit.
    angle_max : float
        We only bin angles from 0 to angle_max.
    num_bins : int
        Number of angle bins from 0 to angle_max.
    error_bars : str
        'sem' for standard error, 'sd' for standard deviation.
    color : tuple
        RGB color for the line plot (0-1 range).
    """

    # 1. Gather data from each session (reaction time + abs(cue_presentation_angle))
    all_data = []  # Will hold (abs_cue_angle, reaction_time)

    for session in sessions_list:
        for trial in session.trials:
            # Must have sensor_touched
            if not trial.get("next_sensor"):
                continue

            # Optionally skip catch trials
            if trial.get("catch", False):
                continue

            # Reaction time
            cue_start = trial.get('cue_start')
            sensor_touch_time = trial["next_sensor"].get('sensor_start')
            if cue_start is None or sensor_touch_time is None:
                continue

            rt = float(sensor_touch_time) - float(cue_start)
            if rt < 0:
                continue  # skip negative RT
            if rt > 5:
                continue

            # Must have turn_data
            turn_data = trial.get("turn_data")
            if not turn_data:
                continue

            # Movement to front port
            port_touched_angle = turn_data.get("port_touched_angle")
            if port_touched_angle is None:
                continue
            if abs(port_touched_angle) > front_limit:
                continue  # Only forward moves

            # Check if trial is incorrect
            if not trial.get("correct_port") or not trial["next_sensor"].get("sensor_touched"):
                continue
            correct_port = int(trial["correct_port"][-1])
            touched_port = int(trial["next_sensor"]["sensor_touched"][-1])
            if correct_port == touched_port:
                continue  # skip correct

            # Use absolute value of cue angle
            cue_angle = turn_data.get("cue_presentation_angle")
            if cue_angle is None:
                continue
            abs_angle = abs(cue_angle)

            # Filter out angles beyond angle_max
            if abs_angle > angle_max:
                continue

            all_data.append((abs_angle, rt))

    # If no data, just print a note and return
    if not all_data:
        print("No data found for the given filtering (incorrect + front-limit + abs angle).")
        return

    # 2. Bin the data by absolute cue angle: from 0 to angle_max
    bin_edges = np.linspace(0, angle_max, num_bins + 1)
    bin_rts = [[] for _ in range(num_bins)]

    for angle, rt in all_data:
        idx = np.searchsorted(bin_edges, angle, side='right') - 1
        idx = min(max(idx, 0), num_bins - 1)
        bin_rts[idx].append(rt)

    print(f"Total trials included after filtering: {len(all_data)}")

    # 3. Compute average RT + error measure
    means = []
    errors = []
    bin_centers = []
    bin_counts = []

    for i in range(num_bins):
        bin_center = 0.5 * (bin_edges[i] + bin_edges[i+1])
        bin_data = bin_rts[i]
        bin_centers.append(bin_center)
        bin_counts.append(len(bin_data))

        if bin_data:
            avg = mean(bin_data)
            if error_bars == 'sem':
                st_err = (stdev(bin_data) / math.sqrt(len(bin_data))) if len(bin_data) > 1 else 0
                errors.append(st_err)
            elif error_bars == 'sd':
                errors.append(stdev(bin_data) if len(bin_data) > 1 else 0)
            else:
                errors.append(0)
            means.append(avg)
        else:
            errors.append(0)
            means.append(float('nan'))

    # Print bin-level stats
    print("Bin Stats (Center | Count | Mean RT):")
    for bc, cnt, m in zip(bin_centers, bin_counts, means):
        rt_str = f"{m:.3f}" if not math.isnan(m) else "N/A"
        print(f"  Bin Center: {bc:.2f}, Count: {cnt}, Mean RT: {rt_str}")

    # 4. Plot with secondary y-axis
    fig, ax_rt = plt.subplots(figsize=(8, 5))

    ax_rt.set_title(title)
    ax_rt.set_xlabel("Absolute Cue Presentation Angle (degrees)")
    ax_rt.set_ylabel("Mean Reaction Time (s)")
    ax_rt.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)

    # Reaction time line
    line_rt = ax_rt.errorbar(
        bin_centers, 
        means, 
        yerr=errors,
        color=color,
        marker='o',
        linestyle='-', 
        capsize=4, 
        ecolor=color, 
        label='Mean RT (Incorrect-Front)'
    )

    # Manual Y-limit for RT, ensuring all error bars visible
    # Collect max of (mean + error) ignoring NaN
    rt_upper = [m + e for m, e in zip(means, errors) if not math.isnan(m)]
    if rt_upper:
        max_rt = max(rt_upper)
        ax_rt.set_ylim(0, max_rt * 1.1)
    else:
        ax_rt.set_ylim(0, 1)

    # Secondary axis for trial count
    ax_count = ax_rt.twinx()
    ax_count.set_ylabel("Number of Trials (n)", color='gray')

    # Plot bin_counts on secondary axis
    line_count = ax_count.plot(
        bin_centers,
        bin_counts,
        color='gray',
        linewidth=2,
        marker='x',
        alpha=0.7,
        zorder=0,
        label='Trial Count'
    )

    # Expand y-limit to fit all counts
    max_cnt = max(bin_counts) if bin_counts else 0
    ax_count.set_ylim(0, max_cnt * 1.1 if max_cnt > 0 else 1)

    # Merge legend
    lns = [line_rt[0], line_count[0]]
    labs = [ln.get_label() for ln in lns]
    ax_rt.legend(lns, labs, loc='best')

    fig.tight_layout()
    plt.show()

    # 5. Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        fname = f"incorrect_rt_vs_abs_cue_angle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(output_path / fname, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path / fname}")
