from typing import List, Dict
from utils.Session_nwb import Session

class SessionManager:
    def __init__(self, cohort):
        self._session_cache: Dict[str, List] = {}
        self.cohort = cohort
    
    def get_sessions(self, session_list: List[str], recalculate=False) -> List:
        """
        Gets or creates session objects based on session IDs.
        
        Args:
            session_list: List of session IDs
            
        Returns:
            List of Session objects
        """
        # Create a cache key based on the session IDs
        cache_key = ','.join(sorted(session_list))
        
        # Check if we already have these sessions cached
        if cache_key in self._session_cache:
            print(f"Using cached sessions for {len(session_list)} sessions")
            return self._session_cache[cache_key]
        
        # If not cached, create the session objects
        print(f"Creating new session objects for {len(session_list)} sessions")
        session_objects = [Session(self.cohort.get_session(session), recalculate) 
                         for session in session_list]
        
        # Cache the results
        self._session_cache[cache_key] = session_objects
        
        return session_objects
    
    def clear_cache(self):
        """Clears the session cache"""
        self._session_cache.clear()
        print("Session cache cleared")

# Example usage:
"""
# Create a session manager for your cohort
march_manager = SessionManager(march_cohort)

# Define your session lists as before
march_cue_group_unlimited = [session for session in march_phases['9'] 
                           if session[:6] == '240323']

# Then use the session manager to get the session objects
march_sessions = march_manager.get_sessions(march_cue_group_unlimited)

# Later in your notebook, if you need the same sessions again,
# they'll be returned from cache instead of being recreated
same_march_sessions = march_manager.get_sessions(march_cue_group_unlimited)

# If you need to clear the cache (e.g., to free memory)
march_manager.clear_cache()
"""