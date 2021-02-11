import sys

from fcutils import path as fcpath
from fcutils.progress import track

sys.path.append("./")
from data_wrangling import paths

"""
    Make bash scripts for running DLC and Control on data
"""

TRACKING_BASH_TEMPLATE = """#! /bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 80G # memory pool for all cores
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -n 10
#SBATCH -t 2-0:0 # time
#SBATCH	-o err.err
#SBATCH -e err.err

echo "loading conda env"
module load miniconda
module load nvidia/9.0

conda activate dlc
export DLClight=True
export CUDA_VISIBLE_DEVICES=1

echo "running tracking"
python /nfs/winstor/branco/Federico/Locomotion/control/LocomotionControl/experimental_validation/dlc_on_hpc.py \\
        /nfs/winstor/branco/Federico/Locomotion/control/experimental_validation/2WDD/Kinematics_FC-FC-2021-01-25/config.yaml \\
        VIDEO \\
        SAVE
"""


def make_dlc_bash_text(video_path, save_path):
    """
        Creates a string with the content of a .sh script for running 
        deeplabcut on HPC
    """
    video_path = (
        str(video_path)
        .replace("Z:\\swc\\", "/nfs/winstor/")
        .replace("\\", "/")
    )
    save_path = (
        str(save_path).replace("Z:\\swc\\", "/nfs/winstor/").replace("\\", "/")
    )

    return TRACKING_BASH_TEMPLATE.replace("VIDEO", video_path).replace(
        "SAVE", save_path
    )


def make_dlc_bash_scripts():
    """
        Generate bash files for tracking all trials videos
    """

    for sub in paths.subfolders.values():
        for video in track(fcpath.files(sub)):
            bash = make_dlc_bash_text(video, paths.tracking_folder)
            bash_path = (
                paths.bash_scripts_folder / f"{sub.name}_{video.stem}.sh"
            )
            with open(bash_path, "w") as out:
                out.write(bash)


if __name__ == "__main__":
    make_dlc_bash_scripts()
