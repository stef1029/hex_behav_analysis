# Cohort Visualizer

A Python package for visualizing and analyzing mouse cohort data with an interactive HTML dashboard.

## Features

- Organize experiment sessions by date and mouse
- Generate interactive visualizations for cohort data analysis
- Track experimental phase progression for individual mice
- Explore session details in an interactive dashboard
- Filter and search through session data with ease

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/cohort-visualizer.git
cd cohort-visualizer

# Install the package
pip install -e .
```

## Quick Start

```python
from cohort_visualizer.core.cohort_date_organizer import CohortDateOrganizer
from cohort_visualizer.visualization.dashboard_generator import DashboardGenerator

# Initialize the cohort organizer with your data directory
cohort_organizer = CohortDateOrganizer(
    "path/to/your/cohort_data",
    multi=True,  # If data is split across multiple mouse folders
    portable_data=True,  # If using portable data format
    use_existing_cohort_info=True  # Use existing cohort_info.json if available
)

# Generate the dashboard
dashboard_generator = DashboardGenerator(cohort_organizer)
dashboard_path = dashboard_generator.generate_dashboard()

# Open the dashboard in a web browser
dashboard_generator.open_dashboard()
```

## Command Line Usage

The package also provides a command-line script for generating dashboards:

```bash
# Generate a dashboard for a cohort directory
python -m scripts.generate_dashboard /path/to/cohort_directory --multi --portable-data

# For help with options
python -m scripts.generate_dashboard --help
```

## Dashboard Features

The interactive dashboard includes:

- **Overview**: Summary statistics and key visualizations of your cohort
- **Timeline**: Heatmap and weekly activity charts showing experiment progression
- **Mouse Analysis**: Individual mouse tracking with phase progression visualization
- **Phase Analysis**: Analyze data by experimental phase
- **Session Explorer**: Advanced filtering and searching of session data

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- plotly (for some visualizations)

## Project Structure

```
cohort_visualizer/
├── core/
│   ├── cohort_date_organizer.py  # Core data organization functionality
│   └── data_processor.py         # Data processing for visualizations
├── visualization/
│   ├── dashboard_generator.py    # Dashboard generation
│   ├── html_templates.py         # HTML templates for the dashboard
│   └── plot_generator.py         # Static plot generation
└── utils/
    └── file_utils.py             # File handling utilities
```

## License

[MIT License](LICENSE)