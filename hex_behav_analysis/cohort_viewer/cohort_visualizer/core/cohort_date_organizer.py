"""
Core module for organizing cohort data by date.
Extends the base Cohort_folder class with date-specific organization methods.
"""

import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Import the Cohort_folder class - adjust this import as needed for your project
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder


class CohortDateOrganizer(Cohort_folder):
    """
    A class that extends Cohort_folder to provide date-based organization and analysis of sessions.
    Inherits all functionality from Cohort_folder and adds methods for organizing sessions by date.
    """
    
    def __init__(
        self,
        cohort_directory,
        multi=True,
        portable_data=False,
        OEAB_legacy=True,
        ignore_tests=True,
        use_existing_cohort_info=False,
        plot=False
    ):
        """
        Initialises the CohortDateOrganizer with the same parameters as Cohort_folder.
        
        Args:
            cohort_directory: Base directory for your data
            multi: Whether the data is split across subfolders (multiple mice) or not
            portable_data: Whether to use the 'portable_data' logic vs. full raw-data logic
            OEAB_legacy: Whether to look for legacy OEAB folder structures
            ignore_tests: Skip any session folders that look like test sessions
            use_existing_cohort_info: If True and cohort_info.json exists, load from it and skip scanning
            plot: Whether to produce a cohort summary plot (if relevant)
        """
        # Initialise the parent class (Cohort_folder)
        super().__init__(
            cohort_directory, 
            multi=multi,
            portable_data=portable_data,
            OEAB_legacy=OEAB_legacy,
            ignore_tests=ignore_tests,
            use_existing_cohort_info=use_existing_cohort_info,
            plot=plot
        )
        
        # Additional initialisations for this class
        self._date_cache = None  # Cache for get_sessions_by_date results
    
    def get_sessions_by_date(self, refresh_cache=False):
        """
        Organises all sessions by date, with sessions for each date grouped by mouse.
        
        Args:
            refresh_cache: Whether to force recalculation even if cached data exists
            
        Returns:
            A dictionary with dates as keys, and for each date a dictionary with mice as keys
            and lists of session IDs as values.
            
            Example structure:
            {
                "2024-03-11": {
                    "mouse1": ["240311_183300_mouse1", "240311_190000_mouse1"],
                    "mouse2": ["240311_185000_mouse2"]
                },
                "2024-03-12": {
                    "mouse1": ["240312_143000_mouse1"]
                }
            }
        """
        # Return cached result if available and refresh not requested
        if self._date_cache is not None and not refresh_cache:
            return self._date_cache
        
        # Dictionary to store sessions organised by date and mouse
        sessions_by_date = defaultdict(lambda: defaultdict(list))
        
        # Iterate through all mice and their sessions
        for mouse_id, mouse_data in self.cohort["mice"].items():
            for session_id in mouse_data["sessions"]:
                # Extract the date from the session ID (first 6 characters: YYMMDD)
                date_str = session_id[:6]
                
                # Convert to YYYY-MM-DD format for better readability and sorting
                try:
                    date_obj = datetime.strptime(date_str, "%y%m%d")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    
                    # Add the session to the appropriate date and mouse
                    sessions_by_date[formatted_date][mouse_id].append(session_id)
                except ValueError:
                    print(f"Warning: Couldn't parse date from session ID {session_id}")
        
        # Sort session lists for each mouse
        for date in sessions_by_date:
            for mouse in sessions_by_date[date]:
                sessions_by_date[date][mouse].sort()
        
        # Convert defaultdict to regular dict for better usability
        result = {date: dict(mice) for date, mice in sessions_by_date.items()}
        
        # Sort the dictionary by date (keys)
        result = dict(sorted(result.items()))
        
        # Cache the result
        self._date_cache = result
        
        return result
    
    def get_date_summary(self):
        """
        Provides a summary of activity by date, showing the number of sessions and mice per date.
        
        Returns:
            A pandas DataFrame with columns:
            - Date: The date in YYYY-MM-DD format
            - Sessions: Total number of sessions on that date
            - Mice: Number of unique mice with sessions on that date
            - Mouse IDs: List of mice IDs with sessions on that date
        """
        sessions_by_date = self.get_sessions_by_date()
        
        summary_data = []
        for date, mice_sessions in sessions_by_date.items():
            total_sessions = sum(len(sessions) for sessions in mice_sessions.values())
            mice_count = len(mice_sessions)
            mice_ids = list(mice_sessions.keys())
            
            summary_data.append({
                "Date": date,
                "Sessions": total_sessions,
                "Mice": mice_count,
                "Mouse IDs": ", ".join(mice_ids)
            })
        
        # Create DataFrame and ensure dates are ordered
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values("Date")
        
        return df
    
    def get_mouse_calendar(self, mouse_id=None):
        """
        Creates a calendar-style representation of experiment days for a specific mouse
        or all mice if mouse_id is None.
        
        Args:
            mouse_id: ID of the mouse to generate calendar for, or None for all mice
            
        Returns:
            A pandas DataFrame with dates as rows and mice as columns.
            Cell values are the number of sessions for that mouse on that date.
        """
        sessions_by_date = self.get_sessions_by_date()
        
        # Prepare data for the calendar
        calendar_data = []
        for date, mice_sessions in sessions_by_date.items():
            row_data = {"Date": date}
            
            # If a specific mouse is requested, only include that mouse
            if mouse_id is not None:
                if mouse_id in mice_sessions:
                    row_data[mouse_id] = len(mice_sessions[mouse_id])
                else:
                    row_data[mouse_id] = 0
            # Otherwise include all mice
            else:
                for m_id in self.cohort["mice"].keys():
                    row_data[m_id] = len(mice_sessions.get(m_id, []))
            
            calendar_data.append(row_data)
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(calendar_data)
        if not df.empty:
            df = df.sort_values("Date")
            df = df.set_index("Date")
        
        return df
    
    def get_date_session_details(self, date):
        """
        Retrieves detailed information about all sessions on a specific date.
        
        Args:
            date: The date to get information for, in YYYY-MM-DD format
            
        Returns:
            A dictionary with mice as keys, and for each mouse a dictionary
            of session details with session IDs as keys.
        """
        sessions_by_date = self.get_sessions_by_date()
        
        # Check if the date exists in our data
        if date not in sessions_by_date:
            print(f"No sessions found for date {date}")
            return {}
        
        result = {}
        for mouse_id, session_ids in sessions_by_date[date].items():
            result[mouse_id] = {}
            
            for session_id in session_ids:
                # Get session details from the cohort information
                session_info = self.get_session(session_id)
                
                # Add key details to the result
                if session_info:
                    # Get phase information depending on data type
                    if self.portable_data:
                        phase = session_info.get("Behaviour_phase")
                    else:
                        phase = session_info.get("raw_data", {}).get("session_metadata", {}).get("phase")
                    
                    # Simplified session details
                    result[mouse_id][session_id] = {
                        "phase": phase,
                        "directory": session_info.get("directory"),
                        "time": session_id[7:13],  # Extract HHMMSS from session ID
                    }
                    
                    # Add additional details if available
                    if not self.portable_data and "raw_data" in session_info:
                        result[mouse_id][session_id].update({
                            "total_trials": session_info["raw_data"]["session_metadata"].get("total_trials"),
                            "video_length": session_info["raw_data"].get("video_length"),
                            "is_complete": session_info["raw_data"].get("is_all_raw_data_present?", False)
                        })
        
        return result
    
    def get_phase_progression_by_mouse(self):
        """
        Tracks the progression of behaviour phases for each mouse over time.
        
        Returns:
            A dictionary with mouse IDs as keys, and for each mouse a list of 
            dictionaries containing date, session_id and phase information.
        """
        progression = {}
        
        # Iterate through mice and their sessions
        for mouse_id, mouse_data in self.cohort["mice"].items():
            mouse_sessions = []
            
            for session_id in mouse_data["sessions"]:
                session_info = self.get_session(session_id)
                
                # Extract date from session ID
                date_str = session_id[:6]
                try:
                    date_obj = datetime.strptime(date_str, "%y%m%d")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    
                    # Get phase information depending on data type
                    if self.portable_data:
                        phase = session_info.get("Behaviour_phase")
                    else:
                        phase = session_info.get("raw_data", {}).get("session_metadata", {}).get("phase")
                    
                    # Add session to the mouse's progression
                    mouse_sessions.append({
                        "date": formatted_date,
                        "session_id": session_id,
                        "phase": phase
                    })
                except ValueError:
                    print(f"Warning: Couldn't parse date from session ID {session_id}")
            
            # Sort sessions by date
            mouse_sessions.sort(key=lambda x: x["date"])
            progression[mouse_id] = mouse_sessions
        
        return progression
    
    def get_session_counts_by_phase(self):
        """
        Counts the number of sessions per behavioural phase.
        
        Returns:
            A dictionary with phases as keys and counts as values.
        """
        phase_counts = defaultdict(int)
        
        # Iterate through all mice and their sessions
        for mouse_id, mouse_data in self.cohort["mice"].items():
            for session_id in mouse_data["sessions"]:
                session_info = self.get_session(session_id)
                
                # Get phase information depending on data type
                if self.portable_data:
                    phase = session_info.get("Behaviour_phase")
                else:
                    phase = session_info.get("raw_data", {}).get("session_metadata", {}).get("phase")
                
                # Increment count for this phase
                phase_counts[phase] += 1
        
        # Convert defaultdict to regular dict
        return dict(phase_counts)
    
    def export_date_summary(self, output_file=None):
        """
        Exports a summary of session activity by date to a CSV file.
        
        Args:
            output_file: Path to the output CSV file. If None, defaults to
                        "date_summary.csv" in the cohort directory.
                        
        Returns:
            Path to the saved file
        """
        summary_df = self.get_date_summary()
        
        if output_file is None:
            output_file = self.cohort_directory / "date_summary.csv"
        
        summary_df.to_csv(output_file, index=False)
        print(f"Date summary exported to {output_file}")
        
        return output_file