"""
Setup script for the cohort_visualizer package.
"""

from setuptools import setup, find_packages

setup(
    name="cohort_visualizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "generate-cohort-dashboard=scripts.generate_dashboard:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for visualizing cohort data",
    keywords="visualization, cohort, dashboard",
    python_requires=">=3.7",
)