import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def graphical_cohort_info(cohort_info, cohort_directory, show=False):
    # Flatten to get session data
    sessions_data = []
    for mouse in cohort_info["mice"]:
        for session_id, session_info in cohort_info["mice"][mouse]["sessions"].items():
            phase = session_info.get("Behaviour_phase", "")
            cue_dur = session_info.get("cue_duration", "")
            wait_dur = session_info.get("wait_duration", "0")
            sessions_data.append({
                "SessionID": session_id,
                "Mouse": mouse,
                "Behaviour_phase": phase,
                "cue_duration": cue_dur,
                "wait_duration": wait_dur
            })

    data = pd.DataFrame(sessions_data)
    data['SessionDate'] = pd.to_datetime(
        data['SessionID'].str[:6],
        format='%y%m%d',
        errors='coerce'
    ).dt.date

    # For each row = 1 session
    data['session_count'] = 1

    session_count = data.pivot_table(
        index='SessionDate',
        columns='Mouse',
        values='session_count',
        aggfunc='sum'
    ).fillna(0)

    # Figure out how many rows (days) and columns (mice) there are
    n_rows = session_count.shape[0]  # number of dates
    n_cols = session_count.shape[1]  # number of mice

    # Customize the size of each cell
    # You can tune these values to get the desired cell size
    cell_height = 1.5
    cell_width = 2

    # Calculate overall figure size (width, height)
    fig_width = n_cols * cell_width
    fig_height = n_rows * cell_height

    # For the color bar ticks
    max_sessions = int(session_count.values.max())
    tick_range = range(0, max_sessions + 1)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        session_count,
        cmap="viridis",
        linewidths=.5,
        cbar=True,
        vmin=0,
        vmax=max_sessions,
        cbar_kws={
            "label": "Number of Sessions",
            "ticks": tick_range
        }
    )
    ax.set_title('Session Count by Mouse and Date')

    # Annotate the cells
    for y in range(session_count.shape[0]):
        for x in range(session_count.shape[1]):
            val = session_count.iloc[y, x]
            if val > 0:
                date = session_count.index[y]
                mouse = session_count.columns[x]
                sessions_for_cell = data[
                    (data['SessionDate'] == date) &
                    (data['Mouse'] == mouse)
                ]

                text_lines = []
                if len(sessions_for_cell) > 1:
                    text_lines.append(f"Sessions: {len(sessions_for_cell)}")

                for _, row in sessions_for_cell.iterrows():
                    phase_text = f"Phase: {row['Behaviour_phase']}"
                    params = []
                    if pd.notna(row['cue_duration']) and str(row['cue_duration']).strip():
                        params.append(f"cue:{row['cue_duration']}ms")
                    if pd.notna(row['wait_duration']) and row['wait_duration'] != "0":
                        params.append(f"wait:{row['wait_duration']}ms")
                    if params:
                        phase_text += f"\n({', '.join(params)})"
                    text_lines.append(phase_text)

                cell_text = '\n'.join(text_lines)
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    cell_text,
                    ha='center',
                    va='center',
                    color='black',
                    fontsize='small',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='none',
                        boxstyle='round,pad=0.2'
                    )
                )

    # Rotate labels, etc.
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = cohort_directory / "cohort_info.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    print(f"\nSaved figure to: {filename}")
