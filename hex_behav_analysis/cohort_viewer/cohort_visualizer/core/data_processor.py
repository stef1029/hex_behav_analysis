"""
Data processor module for transforming cohort data into formats suitable for visualization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict


class CohortDataProcessor:
    """
    Processes cohort data from a CohortDateOrganizer into formats suitable for visualizations.
    """
    
    def __init__(self, cohort_organizer):
        """
        Initialises the data processor with a CohortDateOrganizer instance.
        
        Args:
            cohort_organizer: A CohortDateOrganizer instance containing cohort data
        """
        self.organizer = cohort_organizer
        self.sessions_by_date = self.organizer.get_sessions_by_date()
    
    def get_date_activity_data(self):
        """
        Creates a dataset showing activity over time, suitable for timeline visualizations.
        
        Returns:
            A pandas DataFrame with dates and session counts.
        """
        date_summary = self.organizer.get_date_summary()
        return date_summary[["Date", "Sessions"]]
    
    def get_mouse_heatmap_data(self):
        """
        Creates a dataset for a heatmap showing mouse activity by date.
        
        Returns:
            A pandas DataFrame with dates as rows and mice as columns.
        """
        return self.organizer.get_mouse_calendar()
    
    def get_phase_distribution_data(self):
        """
        Creates a dataset showing the distribution of sessions across different phases.
        
        Returns:
            A pandas DataFrame with phases and their counts.
        """
        phase_counts = self.organizer.get_session_counts_by_phase()
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {"Phase": phase, "Sessions": count} 
            for phase, count in phase_counts.items()
        ])
        
        # Sort by phase if phases are numeric
        try:
            df["Phase"] = pd.to_numeric(df["Phase"], errors="ignore")
            df = df.sort_values("Phase")
        except:
            # If phases are not numeric, leave as is
            pass
        
        return df
    
    def get_mouse_progression_data(self):
        """
        Creates a dataset showing how mice progressed through phases over time.
        
        Returns:
            A pandas DataFrame with mouse ID, date, and phase information.
        """
        progression = self.organizer.get_phase_progression_by_mouse()
        
        # Flatten the progression data for easier plotting
        rows = []
        for mouse_id, sessions in progression.items():
            for session in sessions:
                rows.append({
                    "Mouse": mouse_id,
                    "Date": session["date"],
                    "Phase": session["phase"],
                    "Session ID": session["session_id"]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by mouse and date
        df = df.sort_values(["Mouse", "Date"])
        
        return df
    
    def get_weekly_summary_data(self):
        """
        Creates a dataset summarizing activity by week.
        
        Returns:
            A pandas DataFrame with week numbers and session counts.
        """
        date_summary = self.organizer.get_date_summary()
        
        # Convert date strings to datetime objects
        date_summary["DateTime"] = pd.to_datetime(date_summary["Date"])
        
        # Extract year and week number
        date_summary["Year"] = date_summary["DateTime"].dt.isocalendar().year
        date_summary["Week"] = date_summary["DateTime"].dt.isocalendar().week
        
        # Group by year and week
        weekly_summary = date_summary.groupby(["Year", "Week"]).agg({
            "Sessions": "sum",
            "Mice": "mean",  # Average number of mice per day in the week
            "Date": "count"  # Number of days with sessions in the week
        })
        
        # Reset index for easier usage
        weekly_summary = weekly_summary.reset_index()
        
        # Create a week label (e.g., "2024-W01")
        weekly_summary["WeekLabel"] = weekly_summary.apply(
            lambda row: f"{row['Year']}-W{row['Week']:02d}", axis=1
        )
        
        return weekly_summary
    
    def get_mouse_session_detail_data(self):
        """
        Creates a detailed dataset of all sessions for all mice.
        
        Returns:
            A pandas DataFrame with detailed information for each session.
        """
        rows = []
        
        # Iterate through all dates and mice
        for date, mice_sessions in self.sessions_by_date.items():
            for mouse_id, session_ids in mice_sessions.items():
                for session_id in session_ids:
                    # Get session details
                    session_info = self.organizer.get_session(session_id)
                    
                    # Base row data
                    row = {
                        "Date": date,
                        "Mouse": mouse_id,
                        "Session ID": session_id,
                        "Time": session_id[7:13],  # Extract HHMMSS
                    }
                    
                    # Add phase information
                    if self.organizer.portable_data:
                        row["Phase"] = session_info.get("Behaviour_phase")
                    else:
                        row["Phase"] = session_info.get("raw_data", {}).get(
                            "session_metadata", {}).get("phase")
                    
                    # Add additional details if available
                    if not self.organizer.portable_data and "raw_data" in session_info:
                        row.update({
                            "Total Trials": session_info["raw_data"]["session_metadata"].get("total_trials"),
                            "Video Length (min)": session_info["raw_data"].get("video_length"),
                            "Is Complete": session_info["raw_data"].get("is_all_raw_data_present?", False)
                        })
                    
                    rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by date and time
        if not df.empty:
            df = df.sort_values(["Date", "Time"])
        
        return df
    
    def export_to_json(self, output_dir=None):
        """
        Exports processed data as JSON files for the dashboard.
        
        Args:
            output_dir: Directory to save JSON files. If None, uses 'dashboard_data'
                       subdirectory in the cohort directory.
                       
        Returns:
            Path to the output directory
        """
        if output_dir is None:
            output_dir = self.organizer.cohort_directory / "dashboard_data"
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each dataset as JSON
        datasets = {
            "date_activity.json": self.get_date_activity_data(),
            "mouse_heatmap.json": self.get_mouse_heatmap_data().reset_index(),
            "phase_distribution.json": self.get_phase_distribution_data(),
            "mouse_progression.json": self.get_mouse_progression_data(),
            "weekly_summary.json": self.get_weekly_summary_data(),
            "session_details.json": self.get_mouse_session_detail_data()
        }
        
        for filename, data in datasets.items():
            # Convert to JSON-serializable format
            json_data = data.to_dict(orient="records")
            
            # Write to file
            with open(output_dir / filename, "w") as f:
                json.dump(json_data, f, indent=2)
        
        # Export cohort metadata
        cohort_meta = {
            "cohort_name": self.organizer.cohort.get("Cohort name", "Unknown Cohort"),
            "total_mice": len(self.organizer.cohort.get("mice", {})),
            "total_sessions": sum(
                len(mouse_data.get("sessions", {}))
                for mouse_data in self.organizer.cohort.get("mice", {}).values()
            ),
            "date_range": [
                min(self.sessions_by_date.keys()) if self.sessions_by_date else None,
                max(self.sessions_by_date.keys()) if self.sessions_by_date else None
            ],
            "mice_list": list(self.organizer.cohort.get("mice", {}).keys())
        }
        
        with open(output_dir / "cohort_metadata.json", "w") as f:
            json.dump(cohort_meta, f, indent=2)
        
        print(f"Data exported to {output_dir}")
        return output_dir