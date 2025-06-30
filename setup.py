from setuptools import setup, find_packages

setup(
    name="hex_behav_analysis",
    packages=find_packages(),
    version="0.1",
    install_requires=[
    "opencv-python",
    "pandas", 
    "matplotlib",
    "pynwb",
    "scipy",
    "seaborn",
    "tqdm",
    "ipython",
    
]
)