"""
File utility functions for the cohort visualizer.
"""

import os
import shutil
from pathlib import Path
import json
import tempfile
import webbrowser


def ensure_directory(directory_path):
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data, file_path):
    """
    Saves data as a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    ensure_directory(Path(file_path).parent)
    
    # Convert Path objects to strings for JSON serialization
    if isinstance(data, dict):
        data = {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}
    
    # Write JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return file_path


def load_json(file_path):
    """
    Loads data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Data from the JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def copy_directory_contents(source_dir, target_dir):
    """
    Copies all contents from source directory to target directory.
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        
    Returns:
        Path to the target directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Ensure target directory exists
    ensure_directory(target_path)
    
    # Copy all files and directories
    for item in source_path.glob('*'):
        if item.is_file():
            shutil.copy2(item, target_path / item.name)
        elif item.is_dir():
            shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
    
    return target_path


def create_temp_html(html_content, open_browser=True):
    """
    Creates a temporary HTML file and optionally opens it in the browser.
    
    Args:
        html_content: HTML content as a string
        open_browser: Whether to open the file in the default browser
        
    Returns:
        Path to the temporary HTML file
    """
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.html')
    
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(html_content)
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open(f'file://{path}')
        
        return path
    
    except Exception as e:
        print(f"Error creating temporary HTML file: {e}")
        return None


def open_html_file(file_path):
    """
    Opens an HTML file in the default web browser.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists() and file_path.is_file():
            webbrowser.open(f'file://{file_path.absolute()}')
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    except Exception as e:
        print(f"Error opening HTML file: {e}")
        return False