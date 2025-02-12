# DLCprojectFile = "OutputFolders\wtjx230-3a-11242022163809-0000-OutputFiles\DLC-Project-wtjx230-3a-11242022163809-0000-SRC-2023-02-28"

# create dlcroutine and bash script
# move files to beegfs
# run bash script
import os
import shutil as sh
# takes the project file and original locations of the hal scripts


def editDLCroutine(filetoEdit, newconfig, filetoEditnew = None):
    """
    filetoEdit: orginal DLCroutine.py file
    newconfig: new config path
    filetoEditnew: new file to write to. Default is None, which will overwrite the original file
    """
    with open (filetoEdit, "r") as myfile:
        file = []
        for i, line in list(enumerate(myfile)):
            if i == 0:
                newconfig = newconfig.replace("\\", "/")  # replace \ with / for since linux only takes / in a path
                line = f"config = r'{newconfig}'\n"     # replace config path line.
            file.append(line)
            

    if filetoEditnew is None:       # if no new file is specified, overwrite the original file
        with open(filetoEdit, "w") as myfile:
            for line in file:
                myfile.write(line)
    else:
        # write back to new file:
        with open (filetoEditnew, "w", newline="\n") as myfile:
            # write file to new file
            for line in file:
                myfile.write(line)

def editBashScript(filetoEdit, newDLCroutine, DLCroutineLocation, NumGPUs = 4, filetoEditnew = None):
    """
    filetoEdit: orginal DLCroutine.py file
    newDLCroutine: new DLCroutine.py file geenrated by editDLCroutine
    filetoEditnew: new file to write to. Default is None, which will overwrite the original file
    NumGpus: number of GPUs to use. default is 4
    """
    with open (filetoEdit, "r") as myfile:
        file = []
        for i, line in list(enumerate(myfile)):
            if i == 1:
                line = f"#SBATCH --gres=gpu:{NumGPUs}\n"     # replace gpu number line
            if i == 12:
                line = f'python "{newDLCroutine}"\n'        # replace python script name line
            if i == 10: 
                DLCroutineLocation = DLCroutineLocation.replace("\\", "/")      
                line = f'cd "{DLCroutineLocation}"\n'     # replace cd line to find correct location of python script, inside project file.
            file.append(line)
            print(line)

    if filetoEditnew is None:
        with open(filetoEdit, "w") as myfile:
            # write file to new file
            for line in file:
                myfile.write(line)
    else:
        # write back to new file:
        with open (filetoEditnew, "w", newline = "\n") as myfile:
            # write file to new file
            for line in file:
                myfile.write(line)


import paramiko
def runOnHAL(BashScriptEndLocation, bashScriptName):
    """
    takes location to cd to to find bash script plus name of script to run it.
    """
    host = "hal"
    username = "srogers"
    password = "asDewUrp7"

    client = paramiko.client.SSHClient()    
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    print(BashScriptEndLocation)
    _stdin, _stdout,_stderr = client.exec_command(f'cd "{BashScriptEndLocation}"&& dos2unix "{bashScriptName}"&& sbatch "{bashScriptName}"') # comands to enter into terminal. && means run only if last line worked. ; is normal version.
    print(_stdout.read().decode())  # print any output
    print(_stderr.read().decode()) # print any errors

    client.close()  # close the connection



projectFilePath = r"C:\Users\Stefan R Coltman\OneDrive - University of Cambridge\01 - PhD at LMB\Coding projects\Analysis pipeline\DLC-Project-wtjx230-3a-11242022163809-0000-SRC-2023-02-28"
cephfsLocation  = r"W:\DLC-Projects\Preprocessed"
cephfs2Location = r"V:\DLC-projects\Preprocessed"
beegfs3Location = r"X:\srogers"
DLCRoutinePYpath = r"C:\Users\Stefan R Coltman\OneDrive - University of Cambridge\01 - PhD at LMB\Coding projects\Analysis pipeline\DLCroutine.py"
BashScriptSHpath = r"C:\Users\Stefan R Coltman\OneDrive - University of Cambridge\01 - PhD at LMB\Coding projects\Analysis pipeline\BashScript.sh"

# "C:\Users\Stefan R Coltman\OneDrive - University of Cambridge\01 - PhD at LMB\Coding projects\Analysis pipeline\OutputFolders\wtjx230-3a-11242022163809-0000-OutputFiles\DLC-Project-wtjx230-3a-11242022163809-0000-SRC-2023-02-28"
def RunDLCProject(config_path, NumGPUs = 4):



    # make routine and script files from originals
    editDLCroutine(training_script_path, config_path)
    editBashScript(bash_script_path, training_script_path, HALdlcRoutinePath,  NumGPUs)


    print("Running bash script on hal")
    HALbashFileLocation = os.path.join(beegfs3, projectName).replace("\\", "/")
    # print(HALbashFileLocation)
    runOnHAL(HALbashFileLocation, BashScriptName)
    print("Script complete - check email for confirmation")
    # connect to and run script on hal

if __name__ == "__main__":
    RunDLCProject(projectFilePath, cephfsLocation, beegfs3Location, training_script_path, bash_script_path, NumGPUs = 4)





