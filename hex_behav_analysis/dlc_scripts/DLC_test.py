import deeplabcut
from pathlib import Path

from Preliminary_analysis_scripts.deeplabcut_setup import DLC_setup

videos = str(Path(r"/cephfs2/srogers/New analysis pipeline/training_videos"))


DLC_setup(videos)

