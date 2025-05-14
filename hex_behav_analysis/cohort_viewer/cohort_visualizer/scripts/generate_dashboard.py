"""
Generate dashboard script.

This script creates an interactive HTML dashboard for visualizing cohort data.
"""

import argparse
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cohort_visualizer.core.cohort_date_organizer import CohortDateOrganizer
from cohort_visualizer.visualization.dashboard_generator import DashboardGenerator
from cohort_visualizer.utils.file_utils import open_html_file


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate a cohort dashboard')
    
    parser.add_argument('cohort_directory', type=str, 
                        help='Path to the cohort directory containing session data')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory to save the dashboard (default: cohort_directory/dashboard)')
    
    parser.add_argument('--multi', '-m', action='store_true',
                        help='Whether the data is split across subfolders (multiple mice)')
    
    parser.add_argument('--portable-data', '-p', action='store_true',
                        help='Whether to use portable_data logic vs. full raw-data logic')
    
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open the dashboard in a browser after generation')
    
    parser.add_argument('--use-existing', '-e', action='store_true',
                        help='Use existing cohort_info.json if available')
    
    return parser.parse_args()


def main():
    """
    Main function to generate the dashboard.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Generating dashboard for cohort: {args.cohort_directory}")
    
    # Create the cohort date organizer
    cohort_organizer = CohortDateOrganizer(
        args.cohort_directory,
        multi=args.multi,
        portable_data=args.portable_data,
        use_existing_cohort_info=args.use_existing,
        plot=False  # We don't need the graphical cohort info for the dashboard
    )
    
    # Create the dashboard generator
    dashboard_generator = DashboardGenerator(cohort_organizer)
    
    # Generate the dashboard
    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(args.cohort_directory) / "dashboard"
    
    dashboard_path = dashboard_generator.generate_dashboard(output_dir)
    
    print(f"Dashboard generated at: {dashboard_path}")
    
    # Open the dashboard in a browser if requested
    if not args.no_open:
        print("Opening dashboard in browser...")
        if not open_html_file(dashboard_path):
            print("Failed to open dashboard in browser. Please open manually.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())