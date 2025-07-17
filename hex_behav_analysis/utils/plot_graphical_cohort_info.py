import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_behaviour_data_from_path(file_path):
    """
    Load behaviour data from a specific file path to extract session information.
    
    Args:
        file_path (str): Path to the behaviour data JSON file
        
    Returns:
        dict: Behaviour data dictionary or None if file not found or invalid
    """
    if not file_path:
        return None
        
    behaviour_file = Path(file_path)
    # print(f"Loading behaviour data from: {behaviour_file}")
    
    if behaviour_file.exists():
        try:
            with open(behaviour_file, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def extract_session_info_from_behaviour_data(behaviour_data):
    """
    Extract session information from behaviour data.
    
    Args:
        behaviour_data (dict): Behaviour data dictionary
        
    Returns:
        dict: Dictionary containing phase, cue_duration, wait_duration, and audio_trials
    """
    if behaviour_data is None:
        return {
            "phase": "",
            "cue_duration": "",
            "wait_duration": "",
            "audio_trials": False
        }
    
    # Extract relevant information
    phase = behaviour_data.get("Behaviour phase", "")
    cue_duration = behaviour_data.get("Cue duration", "")
    wait_duration = behaviour_data.get("Wait duration", "")
    
    # Check for audio trials - the key exists only when there are audio trials
    audio_trials = "Audio Trials" in behaviour_data
    
    return {
        "phase": phase,
        "cue_duration": cue_duration,
        "wait_duration": wait_duration,
        "audio_trials": audio_trials
    }

def graphical_cohort_info(cohort_info, cohort_directory, show=False):
    """
    Generate a graphical representation of cohort session information.
    All session details are extracted from behaviour data files.
    
    Args:
        cohort_info (dict): Dictionary containing cohort information
        cohort_directory (Path): Path to the cohort directory
        show (bool): Whether to display the plot
    """
    # Flatten to get session data
    sessions_data = []
    for mouse in cohort_info["mice"]:
        for session_id, session_info in cohort_info["mice"][mouse]["sessions"].items():
            behaviour_file_path = session_info.get("raw_data", {}).get("behaviour_data", "")
            
            # Load behaviour data and extract all information from it
            behaviour_data = load_behaviour_data_from_path(behaviour_file_path)
            session_details = extract_session_info_from_behaviour_data(behaviour_data)
            
            sessions_data.append({
                "SessionID": session_id,
                "Mouse": mouse,
                "Behaviour_phase": session_details["phase"],
                "cue_duration": session_details["cue_duration"],
                "wait_duration": session_details["wait_duration"],
                "audio_trials": session_details["audio_trials"]
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

    # Determine figure dimensions
    n_rows = session_count.shape[0]  # number of dates
    n_cols = session_count.shape[1]  # number of mice

    # Customise the size of each cell
    cell_height = 2.0  # Increased to accommodate more text
    cell_width = 2.5   # Increased for better readability

    # Calculate overall figure size
    fig_width = n_cols * cell_width
    fig_height = n_rows * cell_height

    # For the colour bar ticks
    max_sessions = int(session_count.values.max())
    tick_range = range(0, max_sessions + 1)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        session_count,
        cmap="viridis",
        linewidths=0.5,
        cbar=True,
        vmin=0,
        vmax=max_sessions,
        cbar_kws={
            "label": "Number of Sessions",
            "ticks": tick_range
        },
        annot=False  # Disable default annotation to use custom text
    )
    ax.set_title('Session Count by Mouse and Date')

    # Annotate the cells with detailed information
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
                
                # Add session count if multiple sessions
                if len(sessions_for_cell) > 1:
                    text_lines.append(f"Sessions: {int(val)}")

                for idx, (_, row) in enumerate(sessions_for_cell.iterrows()):
                    # Build session information text
                    session_parts = []
                    
                    # Phase information
                    phase = row['Behaviour_phase']
                    if phase:
                        session_parts.append(f"Phase: {phase}")
                    
                    # Parameter information
                    params = []
                    
                    # Check cue duration
                    cue_dur = row['cue_duration']
                    if pd.notna(cue_dur) and str(cue_dur).strip():
                        cue_str = str(cue_dur).strip()
                        if cue_str and cue_str not in ["0", "0.0", ""]:
                            params.append(f"cue:{cue_str}ms")
                    
                    # Check wait duration
                    wait_dur = row['wait_duration']
                    if pd.notna(wait_dur) and str(wait_dur).strip():
                        wait_str = str(wait_dur).strip()
                        if wait_str and wait_str not in ["0", "0.0", ""]:
                            params.append(f"wait:{wait_str}ms")
                    
                    # Audio trial indicator
                    if row['audio_trials']:
                        params.append("A")
                    
                    # Combine all parts
                    if params:
                        if session_parts:
                            session_text = f"{' '.join(session_parts)} ({', '.join(params)})"
                        else:
                            session_text = f"({', '.join(params)})"
                    else:
                        session_text = ' '.join(session_parts) if session_parts else "No data"
                    
                    # Add session index if multiple sessions
                    if len(sessions_for_cell) > 1:
                        session_text = f"S{idx+1}: {session_text}"
                    
                    text_lines.append(session_text)

                cell_text = '\n'.join(text_lines)
                
                # Calculate font size based on text length and number of lines
                text_length = len(cell_text)
                num_lines = len(text_lines)
                
                if num_lines > 3 or text_length > 80:
                    font_size = 6
                elif num_lines > 2 or text_length > 50:
                    font_size = 7
                elif text_length > 30:
                    font_size = 8
                else:
                    font_size = 9
                
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    cell_text,
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=font_size,
                    weight='bold',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.85,
                        edgecolor='grey',
                        linewidth=0.5,
                        boxstyle='round,pad=0.3'
                    )
                )

    # Format axes
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    filename = cohort_directory / "cohort_info.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    plt.close()

    print(f"\nSaved figure to: {filename}")
    print("Legend: A indicates sessions with audio trials")
    
    # Debug information
    print("\nDebug: Sample session data from behaviour files:")
    sample_count = 0
    for i, row in data.iterrows():
        if sample_count < 3:
            print(f"  Session {row['SessionID']}: phase='{row['Behaviour_phase']}', "
                  f"cue='{row['cue_duration']}', wait='{row['wait_duration']}', "
                  f"audio={row['audio_trials']}")
            sample_count += 1