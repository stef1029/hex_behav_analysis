# Development Environment Setup Guide

This guide explains how to set up your development environment to work with both the July Cohort scripts and your analysis notebooks.

## Prerequisites
- VS Code installed
- Git installed
- Conda installed

## Setup Steps

### 1. Set Up Python Environment

1. Create the Conda Environment
```bash
conda create -n behaviour_analysis python==3.10 -y
```

2. Activate the environment:
```bash
conda activate behaviour_analysis
```

3. Install Dependencies

First, install Python-specific packages:
```bash
python -m pip install open-ephys-python-tools colorama paramiko
```

Then install conda packages:
```bash
conda install matplotlib numpy seaborn scipy opencv h5py
```

Finally, install pynwb:
```bash
pip install pynwb
```

Note: The order of installation is important to ensure proper package compatibility.

### 2. Clone and Set Up Projects
```bash
# Clone the July Cohort scripts repository
git clone [repository-url] July_cohort_scripts

# Create your analysis notebooks folder (if it doesn't exist)
mkdir analysis_notebooks
```

### 3. Install July Cohort Scripts in Editable Mode

Navigate to the July Cohort scripts directory and install it in editable mode:
```bash
cd July_cohort_scripts
pip install -e .
```

This makes the package available to Python while allowing you to edit the source code.

### 4. Configure VS Code Settings

#### For July Cohort Scripts folder:

1. Create a `.vscode` folder and settings.json file:
```bash
mkdir .vscode
```

2. Add this content to `.vscode/settings.json`:
```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ]
}
```

#### For Analysis Notebooks folder:

1. Create the same structure in your notebooks folder:
```bash
cd ../analysis_notebooks
mkdir .vscode
```

2. Add this content to `.vscode/settings.json`:
```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}",
        "C:/Dev/projects/July_cohort_scripts"  # Adjust path as needed
    ]
}
```

### 5. Configure VS Code Python Interpreter

1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Type "Python: Select Interpreter"
4. Choose the `behaviour_analysis` conda environment from the list

### 6. Verify Setup

1. Create a test notebook in your analysis_notebooks folder
2. Try importing a module from the July Cohort scripts
3. If VS Code shows import warnings:
   - Reload VS Code window (`Ctrl+Shift+P` -> "Developer: Reload Window")
   - Or restart the Python Language Server (`Ctrl+Shift+P` -> "Python: Restart Language Server")