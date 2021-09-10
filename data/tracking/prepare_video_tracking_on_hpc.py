from pathlib import Path, PurePosixPath
from loguru import logger

from fcutils.progress import track
from fcutils.path import files

'''
Creates a set of bash scripts to analyze files with DLC on HPC
'''

videos_folder = Path(r'W:\swc\branco\Federico\Locomotion\raw\video')
tracking_folder = Path(r'W:\swc\branco\Federico\Locomotion\raw\tracking')
bash_folder = Path(r'W:\swc\branco\Federico\Locomotion\dlc\dlc_individuals')
winstor_folder = PurePosixPath('nfs/winstor/branco/Federico/Locomotion')

# get videos that have not been tracked yet
videos = files(videos_folder, 'FC_*.avi')
videos = [v for v in videos if '_d' not in v.name and 'test' not in v.name.lower() and 't_' not in v.name]

trackings = [f.name for f in files(tracking_folder, '*.h5')]
logger.info(f'Found {len(videos)} videos and {len(trackings)} tracking files')

to_track = []
for video in videos:
    tracked = [f for f in trackings if video.stem in f]
    if not tracked:
        to_track.append(video)


# prepare bash files
BASH_TEMPLATE = """#! /bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 80G # memory pool for all cores
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -n 10
#SBATCH -t 2-0:0 # time
#SBATCH	-o out.out
#SBATCH -e err.err
#SBATCH --mail-user=federicoclaudi@protonmail.com
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda
module load nvidia/9.0

conda activate dlc
export DLClight=True
export CUDA_VISIBLE_DEVICES=1

echo "running tracking"
python dlc_on_hpc.py '/nfs/winstor/branco/Federico/Locomotion/dlc/locomotion/config.yaml' 'VIDEO' 'DEST'

"""

logger.info(f'Found {len(to_track)} videos left to track. Generating bash files.')
for video in track(to_track):
    winstor_video_path = winstor_folder / 'raw' / 'video' / video.name
    winstor_save_path = winstor_folder / 'raw' / 'tracking'

    bash_content = BASH_TEMPLATE.replace('VIDEO', str(winstor_video_path)).replace('DEST', str(winstor_save_path))

    bash_path = bash_folder/(video.stem+'.sh')
    with open(bash_path, 'w') as fout:
        fout.write(bash_content)

    # dos2unix
    content, outsize = '', 0
    with open(bash_path, 'rb') as infile:
        content = infile.read()
    with open(bash_path, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line+b'\n')

