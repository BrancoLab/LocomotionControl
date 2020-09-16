import sys
from pathlib import Path
import shutil

if sys.platform == "darwin":
    trials_cache = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/m1_cache.h5"

    main_fld = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control"
    )
    frames_cache = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/frames_cache"
    )
else:
    trials_cache = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\behav_data\\m1_cache.h5"

    main_fld = Path(
        "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control"
    )
    frames_cache = Path(
        "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\frames_cache"
    )


# empty frames cahce
try:
    shutil.rmtree(str(frames_cache))
except FileNotFoundError:
    pass
frames_cache.mkdir(exist_ok=True)
