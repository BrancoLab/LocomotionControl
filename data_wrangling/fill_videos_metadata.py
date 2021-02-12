import sys
import os
from pathlib import Path

from fcutils.path import from_json, to_json, files, size
from fcutils.progress import track
from fcutils import video as fcvideo

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent.parent
sys.path.append("./")
sys.path.append(str(module_path))
from data_wrangling import paths


metadata = from_json(paths.metadata_path)
videos_metadata = from_json(paths.videos_metadata_path)

for folder, info in metadata.items():
    for video in track(files(paths.main_folder / folder), description=folder):
        # delete videos that are too small
        if size(video, fmt=False) < 2000:
            print("removing ", video)

        if video.stem in videos_metadata.keys():
            continue

        # try to open video, delete otherwise
        try:
            nframes, width, height, fps, is_color = fcvideo.get_video_params(
                video
            )
        except ValueError:
            print(f"failed to open: {video}")
            video.unlink()
            continue

        video_metadata = info.copy()
        video_metadata["nframes"] = nframes
        video_metadata["width"] = width
        video_metadata["height"] = height
        video_metadata["fps"] = fps
        video_metadata["nframes_before"] = int(fps * info["nsec_before"])
        video_metadata["nframes_after"] = int(fps * info["nsec_after"])
        video_metadata["experiment"] = folder

        videos_metadata[video.stem] = video_metadata

to_json(paths.videos_metadata_path, videos_metadata)
