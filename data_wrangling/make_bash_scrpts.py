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
#SBATCH -n 10
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 2-0:0 # time
#SBATCH	-o out.out
#SBATCH -e err.err

echo "loading conda env"
module load miniconda
module load nvidia/9.0

conda activate dlc
export DLClight=True
export CUDA_VISIBLE_DEVICES=1

echo "running tracking"
python /nfs/winstor/branco/Federico/Locomotion/control/LocomotionControl/experimental_validation/dlc_on_hpc.py \\
        /nfs/winstor/branco/Federico/Locomotion/control/behavioural_data/dlc/alldata-fc-2021-02-11/config.yaml \\
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
    tracked = [
        f.stem.split("DLC")[0]
        for f in fcpath.files(paths.tracking_folder, "*.h5")
    ]
    for sub in paths.subfolders.values():
        for video in track(fcpath.files(sub), description=sub.name):
            if video.stem in tracked:
                continue

            bash = make_dlc_bash_text(video, paths.tracking_folder)
            bash_name = f"{sub.name}_{video.stem}.sh".replace(" ", "")

            bash_path = paths.bash_scripts_folder / bash_name
            with open(bash_path, "w") as out:
                out.write(bash)


if __name__ == "__main__":
    make_dlc_bash_scripts()
