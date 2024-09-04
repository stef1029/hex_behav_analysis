@echo off
SET ENV_NAME=behaviour_analysis

REM Step 1: Create the environment from the YAML file (without filterpy)
echo Creating the Conda environment from environment.yml...
conda env create -f environment.yml

REM Step 2: Activate the environment
echo Activating the environment...
conda activate %ENV_NAME%

REM Step 3: Install filterpy separately using pip
echo Installing filterpy separately using pip...
pip install filterpy future==0.18.3 neo==0.11.1 pillow==5.2.0 quantities==0.13.0

REM Step 4: Verify installation
echo Verifying installation...
conda list filterpy

echo Environment setup complete.