from pathlib import Path
import shutil

from fcutils.progress import track
from fcutils import path

folder = Path(
    r"Z:\swc\branco\Federico\Locomotion\control\behavioural_data\videos for fede"
)
dest_fld = Path(
    r"Z:\swc\branco\Federico\Locomotion\control\behavioural_data\philip"
)


subfolds = (
    "Circle corridor",
    "Circle lights on off (baseline)",
    "Circle void (shelter on side)",
    "Circle void up",
    "Circle wall (non naive)",
    "Circle wall (shelter on side)",
    "Circle wall down (dark non naive)",
    "Circle wall down (delay no baseline)",
    "Circle wall down (no baseline no naive)",
    "Circle wall down (no shelter)",
    "Circle wall down (no baseline)",
    "Circle wall down dark",
    "Circle wall down dark (U shaped)",
    "Circle wall gone unblock",
    "Circle wall lights on off NB",
    "Circle wall up (2)",
    "Circle walls down",
    "Square wall left",
    "Square wall moves left",
    "Square wall moves right",
)


for sub in track(subfolds):
    mice = path.subdirs(folder / sub)
    if not isinstance(mice, list):
        mice = [mice]

    for mouse in track(mice):
        videos = path.files(mouse)
        if videos is None:
            continue
        if not isinstance(videos, list):
            videos = [videos]

        for video in track(videos):
            new_path = dest_fld / video.name
            shutil.move(str(video), str(new_path))
