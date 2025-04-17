from hex_behav_analysis.utils.analysis_manager_arduinoDAQ import *

# Setup:

cohort_directory = Path(r"/cephfs2/dwelch/Behaviour/2501_Lynn_EXCITE")

refresh = False

specific_sessions = []

def main():

    total_start_time = time.perf_counter()
    logger = logging_setup(cohort_directory)
    Cohort = Cohort_folder(cohort_directory, multi = True, plot=False, OEAB_legacy = False)
    directory_info = Cohort.cohort

    # find sessions to process: (if refresh is true then it ignores the processed data flag)
    sessions_to_process = []
    num_sessions = 0
    
    for mouse in directory_info["mice"]:
        for session in directory_info["mice"][mouse]["sessions"]:
            num_sessions += 1
            # session_directory = Path(directory_info["mice"][mouse]["sessions"][session]["directory"])
            if directory_info["mice"][mouse]["sessions"][session]["raw_data"]["is_all_raw_data_present?"] == True:
                if not directory_info["mice"][mouse]["sessions"][session]["processed_data"]["preliminary_analysis_done?"] == True or refresh == True:
                    date = session[:6]
                    if int(date) >= 241001:
                        sessions_to_process.append(Cohort.get_session(session))     # uses .get_session to make sure that analysis manager has all the paths right.

    print(f"Processing {len(sessions_to_process)} of {num_sessions} sessions...")
    # ----------------------

    # if a specific session has been given, then ignore the detectio above:
    if specific_sessions:
        sessions_to_process = [Cohort.get_session(session) for session in specific_sessions]
    # ----------------------

    # run analysis on all sessions in the list:
    for session in sessions_to_process:
        print(f"\n\nProcessing {session.get('directory')}...")
        Process_Raw_Behaviour_Data(session, logger)

    # redo cohort load to refresh jsons:
    Cohort = Cohort_folder(cohort_directory, multi = True, plot=False, OEAB_legacy = False)

    # print total time taken in minutes and seconds, rounded to whole numbers:
    print(f"Total time taken: {round((time.perf_counter() - total_start_time) // 60)} minutes, {round((time.perf_counter() - total_start_time) % 60)} seconds")

if __name__ == "__main__":
    main()
