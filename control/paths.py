import sys
from pathlib import Path

if sys.platform == "darwin":
    trials_cache = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/behav_data/psychometric_trials_augmented.h5"

    main_fld = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion"
    )
    frames_cache = Path(
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/frames_cache"
    )

    db_app = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Apps/loco_upload"

    rnn = (
        "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/RNN"
    )
else:
    trials_cache = Path(
        "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\behav_data\\psychometric_trials_augmented.h5"
    )

    main_fld = Path("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion")
    frames_cache = Path(
        "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\frames_cache"
    )

    db_app = "D:\\Dropbox (UCL - SWC)\\Apps\\loco_upload"

    rnn = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\RNN"


analysis_fld = str(Path(main_fld) / "analysis")

# winstor paths
winstor_main = Path("/nfs/winstor/branco/Federico/Locomotion/control/data")
winstor_trial_cache = "/nfs/winstor/branco/Federico/Locomotion/control/psychometric_trials_augmented.h5"
winstor_rnn = "/nfs/winstor/branco/Federico/Locomotion/control/RNN"
winstor_db_app = ""
