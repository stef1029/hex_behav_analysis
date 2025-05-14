# Cohort Visualizer User Guide

This guide provides detailed instructions on how to use the Cohort Visualizer package to analyze and visualize your experimental cohort data.

## Table of Contents
1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Core Components](#core-components)
4. [The Dashboard](#the-dashboard)
5. [Customization](#customization)
6. [Command Line Usage](#command-line-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Requirements
- Python 3.7 or higher
- pip package manager

### Install from Source
```bash
# Clone the repository (if using git)
git clone https://github.com/yourusername/cohort-visualizer.git
cd cohort-visualizer

# Install the package and dependencies
pip install -e .
```

Or, if you've downloaded the package directly:
```bash
cd path/to/cohort-visualizer
pip install -e .
```

## Getting Started

### Basic Usage
Here's a minimal example to generate a dashboard for your cohort data:

```python
from cohort_visualizer.core.cohort_date_organizer import CohortDateOrganizer
from cohort_visualizer.visualization.dashboard_generator import DashboardGenerator

# Initialize with your data directory
cohort_organizer = CohortDateOrganizer(
    "/path/to/cohort_data",
    multi=True,              # If data is split across multiple mouse folders
    portable_data=True,      # If using portable data format
    use_existing_cohort_info=True  # Use existing cohort_info.json if available
)

# Generate dashboard
dashboard_generator = DashboardGenerator(cohort_organizer)
dashboard_path = dashboard_generator.generate_dashboard()

# Open in browser
dashboard_generator.open_dashboard()
```

### Parameters Explained

The `CohortDateOrganizer` class takes several important parameters:

- `cohort_directory`: The base directory containing your experimental data.
- `multi`: Set to `True` if your data is organized with multiple mice in separate subfolders.
- `portable_data`: Set to `True` if you're using the portable data format rather than full raw data.
- `OEAB_legacy`: Set to `True` if using legacy OEAB folder structures.
- `ignore_tests`: Set to `True` to skip any session folders that look like test sessions.
- `use_existing_cohort_info`: Set to `True` to load from existing JSON files if available.
- `plot`: Set to `True` to generate a summary plot during initialization.

## Core Components

### CohortDateOrganizer

The central class for organizing and accessing your cohort data by date. Key methods include:

- `get_sessions_by_date()`: Organizes all sessions by date and mouse.
- `get_date_summary()`: Provides summary statistics for each date.
- `get_mouse_calendar()`: Creates a calendar-style representation of sessions.
- `get_date_session_details(date)`: Gets detailed information for a specific date.
- `get_phase_progression_by_mouse()`: Tracks phase progression for each mouse.

Example:
```python
# Get sessions organized by date
sessions_by_date = cohort_organizer.get_sessions_by_date()

# Print session information for a specific date
date = "2024-03-15"
if date in sessions_by_date:
    for mouse, sessions in sessions_by_date[date].items():
        print(f"Mouse {mouse} had {len(sessions)} sessions on {date}")
```

### DashboardGenerator

Generates interactive HTML dashboards from your cohort data:

- `generate_dashboard(output_dir=None)`: Creates a complete dashboard.
- `open_dashboard()`: Opens the dashboard in your default web browser.

Example:
```python
# Generate dashboard in a custom location
dashboard_path = dashboard_generator.generate_dashboard("my_dashboard")
print(f"Dashboard saved to {dashboard_path}")
```

### PlotGenerator

Creates static visualizations that can be saved or embedded:

- `create_activity_heatmap()`: Sessions by date and mouse heatmap.
- `create_phase_distribution_barplot()`: Distribution of sessions across phases.
- `create_mouse_progression_plot(mouse_id)`: Phase progression for a specific mouse.
- `create_weekly_activity_plot()`: Session activity by week.

Example:
```python
from cohort_visualizer.visualization.plot_generator import PlotGenerator

# Initialize the plot generator
plot_generator = PlotGenerator(cohort_organizer)

# Generate and save plots
plot_generator.create_activity_heatmap("activity_heatmap.png")
plot_generator.create_weekly_activity_plot("weekly_activity.png")
```

## The Dashboard

The interactive dashboard provides several sections for data exploration:

### Overview Section
- Summary statistics for your cohort
- Session activity chart
- Phase distribution chart

### Timeline Section
- Session heatmap showing activity by date and mouse
- Weekly session activity chart

### Mouse Analysis Section
- Select individual mice for detailed analysis
- Phase progression chart
- Session calendar
- Detailed session table

### Phase Analysis Section
- Analyze sessions by behavioral phase
- Mice distribution by phase
- Phase timeline
- Sessions table for the selected phase

### Session Explorer Section
- Advanced filtering and search capabilities
- Date range selector
- Mouse and phase filters
- Detailed session information

## Customization

### Modifying the Dashboard

The dashboard can be customized by modifying the HTML templates in `cohort_visualizer/visualization/html_templates.py`. 

For example, to change the color scheme:
1. Find the CSS section in the template
2. Modify the colors for phases or other elements
3. Regenerate the dashboard

### Adding Custom Visualizations

To add custom visualizations:
1. Add your custom plot method to `PlotGenerator`
2. Modify the dashboard template to include your new visualization
3. Update the dashboard generation in `DashboardGenerator`

## Command Line Usage

The package comes with a command-line script for easy dashboard generation:

```bash
# Basic usage
python -m scripts.generate_dashboard /path/to/cohort_directory

# With options
python -m scripts.generate_dashboard /path/to/cohort_directory --multi --portable-data --output-dir my_dashboard

# Help
python -m scripts.generate_dashboard --help
```

### Command Line Options
- `cohort_directory`: Path to your cohort data
- `--output-dir`, `-o`: Directory to save the dashboard
- `--multi`, `-m`: Flag for data split across multiple mouse folders
- `--portable-data`, `-p`: Flag for portable data format
- `--no-open`: Don't open the dashboard after generation
- `--use-existing`, `-e`: Use existing cohort_info.json if available

## Troubleshooting

### Common Issues

#### Dashboard not displaying correctly
- Ensure all data files are properly loaded
- Check browser console for JavaScript errors
- Try using a different web browser

#### Missing data in visualizations
- Verify that your data structure matches the expected format
- Check if the cohort_info.json was correctly generated
- Try setting `use_existing_cohort_info=False` to rebuild from scratch

#### Error: "No module named 'cohort_visualizer'"
- Ensure you've correctly installed the package
- Check your Python environment and path
- Try installing in development mode: `pip install -e .`

### Getting Help

If you encounter any issues or have questions, please:
1. Check the documentation in the code comments
2. Review this user guide for solutions
3. Submit an issue on the GitHub repository

## Additional Resources

For more information, see:
- The `scripts/example_usage.py` file for complete working examples
- The README.md file for a quick overview
- The code docstrings for detailed API documentation