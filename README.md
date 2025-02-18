# How to use the Red Hex analysis pipeline:

## Prerequisites:
- ### Data:
  - Collect data on red hex using Main Behaviour Control .py varient. 
    This should create the following files (*SP = Session Prefix. Date-Time_MouseID*):
    - **[SP]_Phase_x_behaviour_data.json**  (Sendkey.py data/ metadata)
    - **[SP]_Tracker_data.json**    (Camera frame ID numbers + IR light tracking data)
    - **[SP]_raw_MP.avi**   (Behaviour Video)
    - **[SP]_Arduino_DAQ.json**     (Arduino based DAQ data/ messages)
    - **[Open Ephys Aquisition Board data folder]** (Ephys data/ sync pulse data)  
  -  These files should all be in a folder with title [SP], and all session folders whould be in one larger cohort folder.

- ### Environment:
  - Access to hex/hal and the cluster required.
  - I use VSCode as my primary analysis environment but there's no reason this can't be done anywhere, eg the terminal or Spyder.
  - The current location of the scripts is: `/cephfs2/srogers/New_analysis_pipeline`
  - Request access to a cpu node with the following command given in hex:
    ```bash
    srun -c112 -n1 --pty bash
    ```
  - Then activate the conda environment BehaviourControl:
    ```bash
    conda activate BehaviourControl
    ```
    - **Note:** This environment will need to be set up before first use on new account.


- ## Preliminary analysis:
    Must be done before any post-processing can happen. 
- ### DAQ processing:
  - **`analysis_manager.py`** is the main script for processing the raw data listed above. It currently requires setting the 
  path to the cohort folder manually within the script at the bottom of it. For faster processing, the main function should be set to main_MP(). Process number also needs changing.
    ```python
    def main_MP():

        cohort_directory = Path(r"/cephfs2/srogers/240207_Dans_data") # Change cohort directory here

        directory_info = Cohort_folder(cohort_directory).cohort
        sessions_to_process = []
        refresh = False
        for mouse in directory_info["mice"]:
            for session in directory_info["mice"][mouse]["sessions"]:
                session_directory = Path(directory_info["mice"][mouse]["sessions"][session]["directory"])
                if directory_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                    if not directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"] == True or refresh == True:
                        sessions_to_process.append(session_directory)
        print(f"Processing {len(sessions_to_process)} sessions...")

        processes = 8       # Change process number to 112 if using cluster, 8 if running on hex/hal

        with mp.Pool(processes) as pool:
            pool.map(Process_Raw_Behaviour_Data, sessions_to_process)

    if __name__ == "__main__":
        main_MP()
    ```
  - **This script will then take about an hour to run.** It performs preliminary processing on the raw data to format the data for faster access by later scripts. A single process version of main() also exists and can be used, however this turns this into an overnight/ day job.
  - **Note:** This script is built to avoid double processing. If a session folder already contains the files that are produced by analysis_manager then these are detected and the session is not added to the list of sessions to process. Therefore this function can safely be run multiple times on one cohort folder and you can add new sessions to is as you like.
  - It produces the following files in each session folder:
    - **[SP]_processed_DAQ_data.json** - This file holds the data that represents the states of components in the red hex rig throughout the session, eg LEDs, solenoids, speakers etc... These values are 1s and 0s in a list which indexes onto another list of timestamps taken from the ephys system, allowing high accuracy syncing of components such as the camera, laser and also the ephys data if in use. 

      *Currently the component data is logged about about 1kHz, however can be pushed to 4kHz with some drop outs of data occasionally*.
    
    - **[SP]_sendkey_logs.json** - Obselete.  Contains a dataframe of session data processed from the inaccurate sendkey logs collected by the computer at the time of behaviour control.
    - **[SP]_video_frame_times.json** - Timestamps for the video frames based on their official frame_ID. Used to account for dropped frames during video recording, so that there is not a drift in video position.
    - **[SP].png** - A saved plot of the DAQ data, allowing a broad raw overview of the session, represented by the states of the rig components throughout.

  - Also created in the cohort folder are cohort summary files which allow easy viewing of the sessions and phases used throughout the cohort. These include:
    - **cohort_info.json** - A detailed breakdown of all file paths and properties of each session, sorted into mice and ordered by date. This dictionary is primarily used by scripts to check for completeness of data and access to specific mice or phases, for example.
    - **concise_cohort_info.json** - As the title suggests, a concise version, designed for easy reading of key info. the purpose of this is largly replaced by cohort_info.png, however.
    - **cohort_info.png** - A graphical representation of key cohort information. This shows the sessions that took place for each of the mice on each day of the training/ recording, as well as the total number of trials that the mouse did that day.

        **Note:** Every time that cohort_folder is run, these files are updated. After adding new files, one run of this is required to see those sessions included in this info.

- ### [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) processing:
  - Each of the videos in the session folders must also be passed through deeplabcut analysis. To achieve this I have pretrained one model on 5 wild-type with no implants performing the behaviour, which serves as the model used on all other similar videos.  
  - Before using the cluster for the first time visit `https://bb8.mrc-lmb.cam.ac.uk/userdash`, login with yuor details, click on slurm link in Additional section and then click 'Go on then' to create an account. If receiving an error like this: `sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified` then you may not have done this already.  

  - The rough overview is that we make a list of videos to analyse, and distribute that across multiple gpu's that we have access to:
    - 1: Run **make_vid_list.py**. Requires you to specify the cohort folder within it. This goes over the cohort and determines which session folders are good to analyse and which have already been analysed. It then produces a .txt in the cohort folder containing this list of directories.
    - 2: In bash, cd to cohort directory, where the videos_to_analyse.txt file is generated by the make_vid_list.py function.
    - 3: Run `NUM_LINES=$(wc -l < videos_to_analyse.txt)`
    - 4: Run `sbatch --array=1-${NUM_LINES}%8 "/cephfs2/srogers/New_analysis_pipeline/Scripts/newSH.sh"`. The %8 in this command specifies the number of gpu's that are to be used.

        This will then run in the terminal and output slurm-out files into the cohort folder. Can take a while.

  - Currently, the model is stored at `config = r'/cephfs2/srogers/New_analysis_pipeline/training_videos/DLC_Project_231212_193535_wtjx285-2a_raw_MP-SRC-2024-01-09/config.yaml'`. This is specified in newSH.sh.


---
# Post-processing:
## Git repo installation and environment setup:

### Analysis Environment Setup

This guide provides instructions to set up the analysis environment for this project using Conda. Follow these steps to clone the repository and set up the environment.

### Prerequisites

Ensure you have the following installed on your machine:
- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)

## Setup Instructions

### 1. Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/your-username/your-repository.git
```

Replace `your-username` and `your-repository` with the actual GitHub username and repository name.

Navigate into the cloned repository directory:

```bash
cd your-repository
```

### 2. Create the Conda Environment

Create the analysis environemt using the following command:  
```bash
conda create -n behaviour_analysis python==3.10 -y
```

### 3. Install dependancies
Activate the environment. Use pip install to add the necessary libraries:
```bash
python -m pip install open-ephys-python-tools colorama paramiko
```
```bash
conda install matplotlib numpy seaborn scipy opencv h5py
```
Then run `pip install pynwb==2.3.3`. This ensure the h5py that it uses works properly maybe?

---

### Troubleshooting

If you encounter any issues during the environment setup
- Ensure that you have the latest version of Conda installed.
- Try uninstalling and reinstalling individual modules. Some can be strange and need you to install via conda to work.

---

## How post-processing works:
  - Post-processing relies primarily on **`Cohort_folder.py`** and **`Session.py`** as the layers through which the session folder data are accessed.
    - `Cohort_folder` usage:
        ```python
        from Cohort_folder import Cohort_folder

        cohort_directory = r"path/to/cohort/session/files"

        cohort = Cohort_folder(cohort_directory) 
        
        info = cohort.cohort  # .cohort accesses the same dictionary as saved in cohort_info.json. This is a very full dictionary and requires a lot of code to find sessions of interest.

        phases = cohort_object.phases   # .phases provides a breakdown of sessions by behaviour phase that was used. This allows easy access to sessions in a particular phase of interest, for example phase 9.

        # For example:
        session_directories = []

        for session in phases["9"]:
            mouse = phases["9"][session]["mouse"]
            if mouse == "WTJP239-4b":
                session_directories.append(Path(phases["9"][session]["path"]))
        ```
    - `Session.py` usage:
        ```python
        from Session import Session

        session_path = r"/cephfs2/srogers/240207_Dans_data/240208_132254_WTJP239-4b"

        session = Session(session_path)

        trials = session.trials
        ```
        
        - Trials are currently formatted as a list of trial dictionaries, each containing this data:
        eg: {'start': 2234722, 'end': 2238854, 'correct_port': 3, 'next_sensor_time': 2238850, 'next_sensor_ID': 3}

        - The numbers here represent indices which allow access to other data forms in a synced up manner.
        For example, the timestamp at which this happened can be accessed though the session.timestamps list:
            
            ```python
            # Calculating trial duration:
            trial_start = session_objects[session_index].timestamps[trials[0]["start"]]
            trial_end = session_objects[session_index].timestamps[trials[0]["next_sensor_time"]]
            trial_time = trial_end - trial_start

            # Evaluating trial success:
            success = True if int(trials[0]["correct_port"]) == int(trials[0]["next_sensor_ID"]) else False
            ```

  - ### Current post-processing scripts:
    All post-processing scripts have the format `PP-[script].py`

    - `PP-plot_performance.py` - Radial plots of performance based on cue presentation angle.
    - `PP-plot_trial_time.py` - Radial plots of time to reach port based on cue presentation angle.
    - `PP-post_processing.py` - Mildly obselete notebook for rough plots of data. Predecessor to Session.py.
    - `PP-sorted_video.py` - Generation of cut videos showing trials rotated and sorted.
    - `Session.draw_LEDs()` - Function for generation of whole session videos with LED locations drawn on. Should really be its own script.



