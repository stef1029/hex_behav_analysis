import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path
from datetime import datetime

from utils.Cohort_folder import Cohort_folder
from utils.Session_nwb import Session

def plot_reaction_time_comparison(
    sessions_list,
    output_path=None,
    title='',
    plot_dist=False,
    draw_stats_lines=True,
    num_bins=50,
    angle_limit=60,
    plot_save_name='untitled_plot',
    draft=True
):
    """
    This function compares reaction times (cue onset to sensor touch) between correct 
    and incorrect trials across sessions, performs a Welch's t-test, and can optionally:
    - Plot distributions on a single axis (overlaid) instead of a box plot.
    - Optionally draw vertical lines for mean and median.

    Parameters:
    -----------
    - sessions_list: List of session objects.
    - output_path: Path to save the output plot (optional).
    - title: Title of the plot.
    - plot_dist: If True, plots distributions overlaid on the same axis.
    - draw_stats_lines: If True, draws vertical lines for mean and median.
    - num_bins: Number of bins to use if plotting distributions.
    - angle_limit: Only include trials where port_touched_angle is within ±angle_limit.
    - plot_save_name: Base name for the saved plot files (default: 'reaction_time_comparison').
    - draft: If True, filenames will be prefixed with datetime, else with 'final'.
    """

    correct_color = (0, 0.68, 0.94)
    incorrect_color = (0.93, 0, 0.55)

    mice_data = {}
    for session in sessions_list:
        mouse_id = session.session_dict.get('mouse_id', 'unknown')
        if mouse_id not in mice_data:
            mice_data[mouse_id] = {'correct': [], 'incorrect': []}

        for trial in session.trials:
            if trial.get("catch", False):
                continue
            
            if not trial.get("next_sensor"):
                continue

            cue_start = trial.get('cue_start')
            sensor_touch_time = trial['next_sensor'].get('sensor_start')
            if cue_start is None or sensor_touch_time is None:
                continue

            reaction_time = float(sensor_touch_time) - float(cue_start)
            if reaction_time < 0:
                continue
            if reaction_time > 5:
                continue

            if "turn_data" not in trial or trial["turn_data"] is None:
                continue

            port_touched_angle = trial["turn_data"].get("port_touched_angle")
            if port_touched_angle is None or not (-angle_limit <= port_touched_angle <= angle_limit):
                continue

            correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
            if correct:
                mice_data[mouse_id]['correct'].append(reaction_time)
            else:
                mice_data[mouse_id]['incorrect'].append(reaction_time)

    correct_rts = [rt for mouse in mice_data.values() for rt in mouse['correct']]
    incorrect_rts = [rt for mouse in mice_data.values() for rt in mouse['incorrect']]

    correct_mean = np.mean(correct_rts) if correct_rts else None
    correct_median = np.median(correct_rts) if correct_rts else None
    incorrect_mean = np.mean(incorrect_rts) if incorrect_rts else None
    incorrect_median = np.median(incorrect_rts) if incorrect_rts else None

    def find_outliers(data):
        if not data:
            return 0, []
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return len(outliers), outliers

    correct_outliers_count, _ = find_outliers(correct_rts)
    incorrect_outliers_count, _ = find_outliers(incorrect_rts)

    print(f"Correct Trials: {len(correct_rts)} included, "
          f"{correct_outliers_count} outliers, "
          f"mean = {correct_mean}, median = {correct_median}")
    print(f"Incorrect Trials: {len(incorrect_rts)} included, "
          f"{incorrect_outliers_count} outliers, "
          f"mean = {incorrect_mean}, median = {incorrect_median}")

    t_stat, p_val = None, None
    if len(correct_rts) > 1 and len(incorrect_rts) > 1:
        t_stat, p_val = stats.ttest_ind(correct_rts, incorrect_rts, equal_var=False)
        print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_val:.3e}")
    else:
        print("Not enough data for statistical test.")

    if plot_dist:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Reaction Time (s)')
        ax.set_ylabel('Density')

        def plot_distribution(ax, data, color, label):
            if not data:
                return
            counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax.plot(bin_centers, counts, color=color, linewidth=2, label=label)

        plot_distribution(ax, correct_rts, correct_color, 'Correct Trials')
        plot_distribution(ax, incorrect_rts, incorrect_color, 'Incorrect Trials')

        if draw_stats_lines:
            if correct_mean is not None:
                ax.axvline(correct_mean, color=correct_color, linestyle='--', label=f'C Mean = {correct_mean:.2f}')
            if correct_median is not None:
                ax.axvline(correct_median, color=correct_color, linestyle='-', label=f'C Median = {correct_median:.2f}')
            if incorrect_mean is not None:
                ax.axvline(incorrect_mean, color=incorrect_color, linestyle='--', label=f'I Mean = {incorrect_mean:.2f}')
            if incorrect_median is not None:
                ax.axvline(incorrect_median, color=incorrect_color, linestyle='-', label=f'I Median = {incorrect_median:.2f}')

        if t_stat is not None and p_val is not None:
            ax.text(
                1.0, 1.02,
                f't = {t_stat:.2f}\np = {p_val:.3e}',
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

        ax.legend(loc='upper right')
        plt.tight_layout()

    else:
        fig = plt.figure(figsize=(3, 6))
        ax = fig.add_subplot(111)
        box_data = [correct_rts, incorrect_rts]
        bp = ax.boxplot(box_data, labels=['Correct', 'Incorrect'],
                        patch_artist=True, showfliers=False, widths=0.6)
        ax.set_title(title)
        ax.set_ylabel('Reaction Time (s)')
        ax.set_xlabel('Trial Type')

        box_colors = [correct_color, incorrect_color]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for medline in bp['medians']:
            medline.set_color('black')
            medline.set_linewidth(2)

        if t_stat is not None and p_val is not None:
            ax.text(
                0.95, 0.95,
                f't = {t_stat:.2f}\np = {p_val:.3e}'
                f'\nMean (C)={correct_mean:.2f}\nMean (I)={incorrect_mean:.2f}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

        plt.tight_layout()

    plt.show()

    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 'dist' if plot_dist else 'box'
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
