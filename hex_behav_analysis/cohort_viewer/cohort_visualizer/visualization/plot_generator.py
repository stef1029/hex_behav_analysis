"""
Plot generator module for creating various visualizations for the dashboard.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import io
import base64


class PlotGenerator:
    """
    Generates static visualizations and plots for embedding in the dashboard.
    Used for generating thumbnails or static images to supplement interactive charts.
    """
    
    def __init__(self, cohort_organizer):
        """
        Initialises the plot generator with a CohortDateOrganizer instance.
        
        Args:
            cohort_organizer: A CohortDateOrganizer instance containing cohort data
        """
        self.organizer = cohort_organizer
        sns.set_style("whitegrid")
    
    def create_activity_heatmap(self, output_path=None, show=False):
        """
        Creates a heatmap of sessions by date and mouse.
        
        Args:
            output_path: Path to save the plot. If None, returns as base64 string.
            show: Whether to display the plot
            
        Returns:
            Path to saved plot or base64 encoded string if output_path is None
        """
        # Get mouse calendar data
        calendar_df = self.organizer.get_mouse_calendar()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            calendar_df, 
            cmap="YlGnBu", 
            annot=True, 
            fmt="d", 
            linewidths=0.5,
            cbar_kws={"label": "Number of Sessions"}
        )
        
        plt.title("Sessions Per Mouse Per Day", fontsize=16)
        plt.xlabel("Mouse ID", fontsize=12)
        plt.ylabel("Date", fontsize=12)
        plt.tight_layout()
        
        # Return plot
        return self._handle_plot_output(plt, output_path, show)
    
    def create_phase_distribution_barplot(self, output_path=None, show=False):
        """
        Creates a bar plot showing the distribution of sessions across different phases.
        
        Args:
            output_path: Path to save the plot. If None, returns as base64 string.
            show: Whether to display the plot
            
        Returns:
            Path to saved plot or base64 encoded string if output_path is None
        """
        # Get phase distribution data
        phase_counts = self.organizer.get_session_counts_by_phase()
        
        # Convert to dataframe
        df = pd.DataFrame([
            {"Phase": phase, "Sessions": count} 
            for phase, count in phase_counts.items()
        ])
        
        # Try to sort numerically if possible
        try:
            df["Phase"] = pd.to_numeric(df["Phase"], errors="ignore")
            df = df.sort_values("Phase")
        except:
            pass
        
        # Define a colormap for phases
        phase_colors = {
            '1': '#007bff',
            '2': '#28a745',
            '3': '#17a2b8',
            '4': '#ffc107',
            '5': '#dc3545',
            '6': '#6610f2',
            '7': '#fd7e14',
            '8': '#20c997',
            '9': '#e83e8c',
            '10': '#6f42c1',
            '3b': '#138496',
            '4b': '#d39e00',
            'test': '#6c757d'
        }
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Get colors for each phase, default to grey if not found
        colors = [phase_colors.get(str(phase), '#6c757d') for phase in df["Phase"]]
        
        sns.barplot(x="Phase", y="Sessions", data=df, palette=colors)
        
        plt.title("Sessions by Behaviour Phase", fontsize=16)
        plt.xlabel("Phase", fontsize=12)
        plt.ylabel("Number of Sessions", fontsize=12)
        plt.tight_layout()
        
        # Return plot
        return self._handle_plot_output(plt, output_path, show)
    
    def create_mouse_progression_plot(self, mouse_id, output_path=None, show=False):
        """
        Creates a plot showing the progression of phases for a specific mouse over time.
        
        Args:
            mouse_id: ID of the mouse to plot
            output_path: Path to save the plot. If None, returns as base64 string.
            show: Whether to display the plot
            
        Returns:
            Path to saved plot or base64 encoded string if output_path is None
        """
        # Get progression data for this mouse
        progression = self.organizer.get_phase_progression_by_mouse()
        
        if mouse_id not in progression:
            print(f"No data found for mouse {mouse_id}")
            return None
        
        mouse_data = progression[mouse_id]
        
        # Convert to dataframe
        df = pd.DataFrame(mouse_data)
        
        # Try to convert phase to numeric for better plotting
        try:
            df["phase_numeric"] = pd.to_numeric(df["phase"], errors="coerce")
        except:
            df["phase_numeric"] = df["phase"]
        
        # Sort by date
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Define a colormap for phases
        phase_colors = {
            '1': '#007bff',
            '2': '#28a745',
            '3': '#17a2b8',
            '4': '#ffc107',
            '5': '#dc3545',
            '6': '#6610f2',
            '7': '#fd7e14',
            '8': '#20c997',
            '9': '#e83e8c',
            '10': '#6f42c1',
            '3b': '#138496',
            '4b': '#d39e00',
            'test': '#6c757d'
        }
        
        # Plot as scatter plot with lines
        for phase in df["phase"].unique():
            if pd.isna(phase):
                continue
            
            phase_df = df[df["phase"] == phase]
            color = phase_colors.get(str(phase), '#6c757d')
            
            plt.scatter(
                phase_df["date"], 
                phase_df["phase_numeric"],
                label=f"Phase {phase}",
                s=100,
                color=color
            )
        
        # Connect points with lines
        plt.plot(df["date"], df["phase_numeric"], color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f"Phase Progression for Mouse {mouse_id}", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Phase", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Return plot
        return self._handle_plot_output(plt, output_path, show)
    
    def create_weekly_activity_plot(self, output_path=None, show=False):
        """
        Creates a line plot showing session activity by week.
        
        Args:
            output_path: Path to save the plot. If None, returns as base64 string.
            show: Whether to display the plot
            
        Returns:
            Path to saved plot or base64 encoded string if output_path is None
        """
        # Get date summary
        date_summary = self.organizer.get_date_summary()
        
        # Convert to weekly data
        date_summary["Date"] = pd.to_datetime(date_summary["Date"])
        weekly_data = date_summary.resample("W-MON", on="Date").sum().reset_index()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(weekly_data["Date"], weekly_data["Sessions"], marker='o', 
                 linestyle='-', linewidth=2, markersize=8, color='#17a2b8')
        
        plt.fill_between(weekly_data["Date"], weekly_data["Sessions"], 
                         alpha=0.2, color='#17a2b8')
        
        plt.title("Weekly Session Activity", fontsize=16)
        plt.xlabel("Week", fontsize=12)
        plt.ylabel("Number of Sessions", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Return plot
        return self._handle_plot_output(plt, output_path, show)
    
    def _handle_plot_output(self, plt_obj, output_path, show):
        """
        Helper method to handle different output options for plots.
        
        Args:
            plt_obj: Matplotlib pyplot object
            output_path: Path to save the plot or None
            show: Whether to display the plot
            
        Returns:
            Path to saved plot or base64 encoded string if output_path is None
        """
        if output_path:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            plt_obj.savefig(output_path, dpi=100, bbox_inches='tight')
            
            if show:
                plt_obj.show()
            else:
                plt_obj.close()
                
            return output_path
        else:
            # Convert to base64 string
            buffer = io.BytesIO()
            plt_obj.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if show:
                plt_obj.show()
            else:
                plt_obj.close()
                
            return f"data:image/png;base64,{image_base64}"