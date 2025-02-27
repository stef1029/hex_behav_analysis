
import shutil as sh
import os
import json
from pathlib import Path
import deeplabcut as dlc
import ruamel.yaml


class DLC_setup():
    def __init__(self, data_folder_path):
        self.data_folder_path = Path(data_folder_path)
        self.session_videos = [file for file in self.data_folder_path.glob('*') if file.suffix == '.avi']

        self.copy_cluster_scripts()

        self.prep_DLC_project_file()

        self.extractFrames()

    def copy_cluster_scripts(self):
        """
        create deeplabcut routine files for running program on HAL - 
        not edited yet, just copies from pipelineFiles folder and gives unique name.
        """
        
        # create unique names for files
        self.DLC_routine_name = f"DLCroutine_{str(self.session_videos)[:-4]}.py"
        self.BASH_script_name = f"BashScript_{str(self.session_videos)[:-4]}.sh"

        # Uses static locations for example scripts, held within analysis pipeline folder.
        example_bash_script_path = Path(r"/cephfs2/srogers/New analysis pipeline/BashScript.sh")
        example_training_script_path = Path(r"/cephfs2/srogers/New analysis pipeline/deeplabcut_training.py")

        if not os.path.exists(os.path.join(self.data_folder_path, self.DLC_routine_name)):
            sh.copy(example_training_script_path, self.data_folder_path)

        if not os.path.exists(os.path.join(self.data_folder_path, self.BASH_script_name)):
            sh.copy(example_bash_script_path, self.data_folder_path)


    def prep_DLC_project_file(self):
        """
        Create new DLC project file for single session video.
        """

        self.project_name = f"DLC_Project_{self.session_videos[0].stem}"

        self.DLC_config_path = dlc.create_new_project(self.project_name, 
                                                      "SRC", 
                                                      self.session_videos, 
                                                      working_directory = self.data_folder_path, 
                                                      copy_videos = True)

        print(f"DLC project file created at {self.DLC_config_path}.")
        
        self.edit_YAML()

    def edit_YAML(self, editOriginal = True):
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with open(self.DLC_config_path, 'r') as stream:
            config = yaml.load(stream)

        config["bodyparts"] = ["nose", "head", "left_ear", "right_ear", "spine_1", "spine_2", "spine_3", "spine_4"]
        config["skeleton"] = [["nose", "head"], ["head", "left_ear"], ["head", "right_ear"], ["head", "spine_1"], 
                              ["spine_1", "spine_2"], ["spine_2", "spine_3"], ["spine_3", "spine_4"]]
        config["numframes2pick"] = 20
        config["move2corner"] = False
        
        if editOriginal:
            output_path = self.DLC_config_path
        else:
            output_path = self.DLC_config_path.replace(".yaml", "_new.yaml")

        with open(output_path, 'w') as stream:
            yaml.dump(config, stream)

    def extractFrames(self):
        """
        Extract frames from videos and cluster them using kmeans
        """
        cluster_step = 10 # consider only every 10th frame for the kmeans
        dlc.extract_frames(self.DLC_config_path, 
                           algo='kmeans', 
                           userfeedback=False, 
                           cluster_step=cluster_step , 
                           cluster_resizewidth=30, 
                           slider_width=25)

    
        

        
