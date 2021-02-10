from pathlib import Path
import shutil

from fcutils.progress import track
from fcutils import path

folder = Path(r"D:\Dropbox (UCL)\Rotation_context\experiments\dwm\data")
dest_fld = Path(
    r"Z:\swc\branco\Federico\Locomotion\control\behavioural_data\matt"
)


for sub in track(path.subdirs(folder)):
    mice = path.subdirs(sub)
    if mice is None or not mice:
        continue
    if not isinstance(mice, list):
        mice = [mice]

    for mouse in mice:
        recs = path.subdirs(mouse)
        if recs is None or not recs:
            continue
        if not isinstance(recs, list):
            recs = [recs]

        for rec in recs:
            videos = path.files(rec, "cam1_FEC*")
            if videos is None or not videos:
                continue
            if not isinstance(videos, list):
                videos = [videos]

            for n, video in enumerate(videos):
                if "avi" in video.name:
                    ext = "avi"
                else:
                    ext = "mp4"
                new_path = (
                    dest_fld / f"{sub.name}_{mouse.name}_{rec.name}_{n}.{ext}"
                )

                if new_path.exists():
                    continue

                if path.size(video, fmt=False) > 5000:
                    continue

                shutil.copy(str(video), str(new_path))
