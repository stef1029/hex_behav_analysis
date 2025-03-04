import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import math
from statistics import mean, stdev

from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session

def plot_incorrect_rt_vs_cue_angle_abs(
    sessions_list,
    output_path=None,
    title="",
    front_limit=60,
    angle_max=180,
    num_bins=12,
    error_bars='sem',
    color=(0.93, 0, 0.55),
    plot_save_name='untitled_plot',
    draft=True
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
    plot_save_name : str
        Base name for the saved plot files.
    draft : bool
        If True, filenames will be prefixed with datetime, otherwise with 'final'.
    """

    all_data = []

    for session in sessions_list:
        for trial in session.trials:
            if not trial.get("next_sensor"):
                continue
            if trial.get("catch", False):
                continue

            cue_start = trial.get('cue_start')
            sensor_touch_time = trial["next_sensor"].get('sensor_start')
            if cue_start is None or sensor_touch_time is None:
                continue

            rt = float(sensor_touch_time) - float(cue_start)
            if rt < 0:
                continue
            if rt > 5:
                continue

            turn_data = trial.get("turn_data")
            if not turn_data:
                continue

            port_touched_angle = turn_data.get("port_touched_angle")
            if port_touched_angle is None:
                continue
            if abs(port_touched_angle) > front_limit:
                continue

            if not trial.get("correct_port") or not trial["next_sensor"].get("sensor_touched"):
                continue
            correct_port = int(trial["correct_port"][-1])
            touched_port = int(trial["next_sensor"]["sensor_touched"][-1])
            if correct_port == touched_port:
                continue

            cue_angle = turn_data.get("cue_presentation_angle")
            if cue_angle is None:
                continue
            abs_angle = abs(cue_angle)
            if abs_angle > angle_max:
                continue

            all_data.append((abs_angle, rt))

    if not all_data:
        print("No data found for the given filtering (incorrect + front-limit + abs angle).")
        return

    bin_edges = np.linspace(0, angle_max, num_bins + 1)
    bin_rts = [[] for _ in range(num_bins)]

    for angle, rt in all_data:
        idx = np.searchsorted(bin_edges, angle, side='right') - 1
        idx = min(max(idx, 0), num_bins - 1)
        bin_rts[idx].append(rt)

    print(f"Total trials included after filtering: {len(all_data)}")

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

    print("Bin Stats (Center | Count | Mean RT):")
    for bc, cnt, m in zip(bin_centers, bin_counts, means):
        rt_str = f"{m:.3f}" if not math.isnan(m) else "N/A"
        print(f"  Bin Center: {bc:.2f}, Count: {cnt}, Mean RT: {rt_str}")

    fig, ax_rt = plt.subplots(figsize=(8, 5))
    ax_rt.set_title(title)
    ax_rt.set_xlabel("Absolute Cue Presentation Angle (degrees)")
    ax_rt.set_ylabel("Mean Reaction Time (s)")
    ax_rt.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)

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

    rt_upper = [m + e for m, e in zip(means, errors) if not math.isnan(m)]
    if rt_upper:
        max_rt = max(rt_upper)
        ax_rt.set_ylim(0, max_rt * 1.1)
    else:
        ax_rt.set_ylim(0, 1)

    ax_count = ax_rt.twinx()
    ax_count.set_ylabel("Number of Trials (n)", color='gray')
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

    max_cnt = max(bin_counts) if bin_counts else 0
    ax_count.set_ylim(0, max_cnt * 1.1 if max_cnt > 0 else 1)

    lns = [line_rt[0], line_count[0]]
    labs = [ln.get_label() for ln in lns]
    ax_rt.legend(lns, labs, loc='best')

    fig.tight_layout()
    plt.show()

    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 'plot'
        if draft:
            base_filename = f"{date_time}_{plot_save_name}_{suffix}"
        else:
            base_filename = f"final_{plot_save_name}_{suffix}"

        output_filename_svg = f"{base_filename}.svg"
        output_filename_png = f"{base_filename}.png"

        counter = 0
        while (output_path / output_filename_svg).exists() or (output_path / output_filename_png).exists():
            counter += 1
            output_filename_svg = f"{base_filename}_{counter}.svg"
            output_filename_png = f"{base_filename}_{counter}.png"

        print(f"Saving plot as SVG to: '{output_path / output_filename_svg}'")
        fig.savefig(output_path / output_filename_svg, format='svg', bbox_inches='tight', transparent=True)

        print(f"Saving plot as PNG to: '{output_path / output_filename_png}'")
        fig.savefig(output_path / output_filename_png, format='png', bbox_inches='tight', transparent=True)
