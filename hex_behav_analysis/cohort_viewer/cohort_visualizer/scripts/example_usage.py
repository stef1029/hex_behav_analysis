"""
Example usage of the cohort_visualizer package.

This script demonstrates how to use the cohort_visualizer package to generate
an interactive dashboard for your cohort data.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cohort_visualizer.core.cohort_date_organizer import CohortDateOrganizer
from cohort_visualizer.visualization.dashboard_generator import DashboardGenerator
from cohort_visualizer.visualization.plot_generator import PlotGenerator


def main():
    """
    Example of how to use the cohort_visualizer package.
    """
    # Replace with your actual cohort directory
    cohort_directory = Path("D:/Behaviour/July_cohort_24/Portable_data")
    
    print(f"Initializing cohort organizer for: {cohort_directory}")
    
    # Initialize the cohort organizer
    cohort_organizer = CohortDateOrganizer(
        cohort_directory,
        multi=True,              # Data is split across multiple mouse folders
        portable_data=True,      # Using portable data format
        use_existing_cohort_info=True,  # Use existing cohort_info.json if available
        plot=False               # Don't create the default cohort info plot
    )
    
    # Example 1: Get and print date summary
    print("\nExample 1: Date Summary")
    date_summary = cohort_organizer.get_date_summary()
    print(date_summary.head())
    
    # Example 2: Get and print mouse calendar
    print("\nExample 2: Mouse Calendar")
    mouse_calendar = cohort_organizer.get_mouse_calendar()
    print(mouse_calendar.head())
    
    # Example 3: Get information for a specific date
    print("\nExample 3: Sessions on a specific date")
    # Get the first date in the data
    first_date = cohort_organizer.get_sessions_by_date().keys()
    if first_date:
        first_date = list(first_date)[0]
        print(f"Sessions on {first_date}:")
        sessions = cohort_organizer.get_date_session_details(first_date)
        for mouse_id, mouse_sessions in sessions.items():
            print(f"  Mouse: {mouse_id}")
            for session_id, session_details in mouse_sessions.items():
                print(f"    - {session_id}: Phase {session_details.get('phase')}")
    
    # Example 4: Generate some static plots
    print("\nExample 4: Generating static plots")
    plot_generator = PlotGenerator(cohort_organizer)
    
    # Create a directory for the plots
    plots_dir = Path("example_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Generate activity heatmap
    print("  Generating activity heatmap...")
    plot_generator.create_activity_heatmap(
        output_path=plots_dir / "activity_heatmap.png", 
        show=False
    )
    
    # Generate phase distribution plot
    print("  Generating phase distribution plot...")
    plot_generator.create_phase_distribution_barplot(
        output_path=plots_dir / "phase_distribution.png", 
        show=False
    )
    
    # Generate weekly activity plot
    print("  Generating weekly activity plot...")
    plot_generator.create_weekly_activity_plot(
        output_path=plots_dir / "weekly_activity.png", 
        show=False
    )
    
    print(f"  Static plots saved to: {plots_dir.absolute()}")
    
    # Example 5: Generate the interactive dashboard
    print("\nExample 5: Generating interactive dashboard")
    dashboard_generator = DashboardGenerator(cohort_organizer)
    
    # Generate dashboard in a custom directory
    dashboard_dir = Path("example_dashboard")
    dashboard_path = dashboard_generator.generate_dashboard(dashboard_dir)
    
    print(f"Dashboard generated at: {dashboard_path}")
    print("Opening dashboard in web browser...")
    
    # Open the dashboard in the default browser
    dashboard_generator.open_dashboard()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()