"""
Dashboard generator module for creating interactive HTML dashboards from cohort data.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import shutil

from ..core.data_processor import CohortDataProcessor
from .html_templates import generate_html_template


class DashboardGenerator:
    """
    Generates an interactive HTML dashboard for visualizing cohort data.
    """
    
    def __init__(self, cohort_organizer):
        """
        Initialises the dashboard generator with a CohortDateOrganizer instance.
        
        Args:
            cohort_organizer: A CohortDateOrganizer instance containing cohort data
        """
        self.organizer = cohort_organizer
        self.processor = CohortDataProcessor(cohort_organizer)
        self.output_dir = None
    
    def generate_dashboard(self, output_dir=None):
        """
        Generates a complete dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save the dashboard. If None, uses 'dashboard'
                       subdirectory in the cohort directory.
                       
        Returns:
            Path to the generated dashboard HTML file
        """
        if output_dir is None:
            output_dir = self.organizer.cohort_directory / "dashboard"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export data as JSON files
        data_dir = self.processor.export_to_json(self.output_dir / "dashboard_data")
        
        # 2. Generate HTML file
        html_path = self.generate_html_file()
        
        # 3. Copy any additional assets (CSS, etc.)
        self.copy_assets()
        
        print(f"Dashboard generated at {html_path}")
        return html_path
    
    def generate_html_file(self):
        """
        Generates the main HTML file for the dashboard.
        
        Returns:
            Path to the generated HTML file
        """
        # Load cohort metadata
        with open(self.output_dir / "dashboard_data" / "cohort_metadata.json", "r") as f:
            cohort_meta = json.load(f)
        
        # Generate date string
        date_generated = datetime.now().strftime("%d %b %Y, %H:%M")
        
        # Generate HTML from template
        html_content = generate_html_template(
            cohort_meta, 
            date_generated,
            self.organizer.cohort_directory
        )
        
        # Write HTML file
        html_path = self.output_dir / "index.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path
    
    def copy_assets(self):
        """
        Copies any additional assets needed for the dashboard.
        This could include CSS files, images, or custom JavaScript.
        """
        # Create assets directory if not exists
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Example: copy a custom CSS file if it exists
        # This is a placeholder - in a real implementation,
        # you would have actual asset files to copy
        custom_css_path = Path(__file__).parent / "assets" / "custom.css"
        if custom_css_path.exists():
            shutil.copy(custom_css_path, assets_dir / "custom.css")
    
    def open_dashboard(self):
        """
        Opens the dashboard in the default web browser.
        
        Returns:
            True if successful, False otherwise
        """
        import webbrowser
        
        dashboard_path = self.output_dir / "index.html"
        if dashboard_path.exists():
            webbrowser.open(f"file://{dashboard_path.absolute()}")
            return True
        else:
            print(f"Dashboard not found at {dashboard_path}")
            return False