import sys

from scipy.signal import medfilt
import pandas as pd
from pathlib import Path

sys.path.append("./")


N_frames = 411763  # same number as number of frames in trials

from data.dbase.db_tables import (
    Tracking,
    SessionCondition,
    LocomotionBouts,
)

save_path = (
    Path(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior"
    )
    / "random_frames_tracking.json"
)

tracking = {"x": [], "y": [], "u": [], "omega": []}

bouts = pd.DataFrame(
    (
        LocomotionBouts * SessionCondition
        & 'complete="true"'
        & 'direction="outbound"'
    ).fetch()
)

sessions = bouts["session_id"].unique()

for sess in sessions:
    # get tracking data
    tracking_data = Tracking.get_session_tracking(
        sess, movement=False, body_only=True
    )

    # append trackingdata to tracking
    for key in tracking_data.keys():
        tracking[key].extend(medfilt(tracking_data[key], 11))

print(f"Got {len(tracking['x'])} tracking data frame")

# randomly sample frames
tracking = pd.DataFrame(tracking)
tracking = tracking.sample(N_frames)

# save to json at save_path
tracking.to_json(save_path)
