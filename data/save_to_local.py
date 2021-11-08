import sys
from tpd import recorder
import time
import pandas as pd
from loguru import logger
import numpy as np

from fcutils.progress import track

sys.path.append("./")


from data.dbase.db_tables import ROICrossing, LocomotionBouts, Tracking
from data import arena


recorder.start(
    base_folder=r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior",
    folder_name="saved_data",
    timestamp=False,
)

# ---------------------------------------------------------------------------- #
#                               download tracking                              #
# ---------------------------------------------------------------------------- #


tracking = {}
sessions = list(set(ROICrossing().fetch("name")))
logger.info(f"Getting tracking data for {len(sessions)} sessions")
for n, session in track(enumerate(sessions), total=len(sessions)):
    tracking[session] = Tracking.get_session_tracking(
        session, movement=False, body_only=False
    )
    time.sleep(0.5)

    # if n > 2:
    #     break

# ---------------------------------------------------------------------------- #
#                              save roi crossings                              #
# ---------------------------------------------------------------------------- #
for ROI in arena.ROIs_dict.keys():
    # fetch from database
    crossings = pd.DataFrame(
        (
            ROICrossing * ROICrossing.InitialCondition
            & f'roi="{ROI}"'
            & "mouse_exits=1"
        ).fetch()
    )
    logger.info(f"Loaded {len(crossings)} crossings for: '{ROI}'")

    roi_crossing_dict = []
    for i, crossing in track(crossings.iterrows(), total=len(crossings)):
        crossing_trk = ROICrossing.get_crossing_tracking(
            crossing, tracking[crossing["name"]]
        )
        roi_crossing_dict.append({**crossing.to_dict(), **crossing_trk})

    recorder.add_data(crossings, f"{ROI}_crossings", fmt="h5")
    recorder.add_data(roi_crossing_dict, f"{ROI}_crossings", fmt="json")


# ---------------------------------------------------------------------------- #
#                             save locomotor bouts                             #
# ---------------------------------------------------------------------------- #
logger.info("Fetching bouts")
bouts = pd.DataFrame(
    (LocomotionBouts & 'complete="true"' & 'direction="outbound"').fetch()
)
logger.info(f"Got {len(bouts)} bouts")


bouts_dicts = []
for i, bout in track(bouts.iterrows(), total=len(bouts)):
    trk = LocomotionBouts.get_bout_tracking(bout, tracking[bout["name"]])
    trk = {bp: np.array(list(v.values())) for bp, v in trk.to_dict().items()}

    bout_dict = {**bout.to_dict(), **trk}
    for k, v in bout_dict.items():
        if isinstance(v, np.ndarray):
            bout_dict[k] = list(v)

    bouts_dicts.append(bout_dict)

recorder.add_data(bouts_dicts, f"complete_bouts", fmt="json")
recorder.describe()
