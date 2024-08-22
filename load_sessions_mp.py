import multiprocessing
from Session_nwb import Session  # Adjust this import based on where your Session class is located

def load_session(session_id, cohort):
    """
    Helper function to load a session object.
    
    Args:
        session_id (str): The session ID.
        cohort (Cohort): The cohort object to use for loading the session.
    
    Returns:
        Session: A Session object.
    """
    return Session(cohort.get_session(session_id))

def load_sessions_in_parallel(session_list, exclusion_list, cohort, num_workers=None):
    """
    Load a list of Session objects in parallel using multiprocessing.
    
    Args:
        session_list (list): List of session IDs to load.
        exclusion_list (list): List of session IDs to exclude.
        cohort (Cohort): The cohort object to use for loading the sessions.
        num_workers (int, optional): Number of worker processes to use. Defaults to None, which uses os.cpu_count().
    
    Returns:
        list: A list of Session objects.
    """
    # Filter out excluded sessions
    filtered_session_list = [session for session in session_list if session not in exclusion_list]
    
    # Use multiprocessing to load sessions in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        session_objects = pool.starmap(load_session, [(session_id, cohort) for session_id in filtered_session_list])
    
    return session_objects
