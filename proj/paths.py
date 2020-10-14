import sys
from pathlib import Path

if sys.platform == "darwin":
    trials_cache = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/m46_cache.h5"
    # trials_cache = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/zanes.h5"

    main_fld = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control"
    )
    frames_cache = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/frames_cache"
    )

    db_app = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Apps/loco_upload"

    rnn_trainig = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/RNN/training_data"
else:
    trials_cache = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\behav_data\\m46_cache.h5"
    # trials_cache = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\behav_data\\zanes.h5"

    main_fld = Path(
        "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control"
    )
    frames_cache = Path(
        "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\frames_cache"
    )

    db_app = "D:\\Dropbox (UCL - SWC)\\Apps\\loco_upload"

    rnn_trainig = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\RNN\\training_data"


analysis_fld = str(Path(main_fld) / "analysis")

# winstor paths
winstor_main = Path("/nfs/winstor/branco/Federico/Locomotion/control/data")
winstor_trial_cache = (
    "/nfs/winstor/branco/Federico/Locomotion/control/m46_cache.h5"
)

# winstor_trial_cache = (
#     "/nfs/winstor/branco/Federico/Locomotion/control/zanes.h5"
# )

winstor_db_app = ""
