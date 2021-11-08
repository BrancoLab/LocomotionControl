import sys
from pathlib import Path
from tpd import recorder
import time
import pandas as pd
from loguru import logger
from fcutils.maths import derivative
import numpy as np

from fcutils.progress import track
sys.path.append("./")


from data.dbase.db_tables import ROICrossing, LocomotionBouts, Tracking
from data import arena


recorder.start(
    base_folder = r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior',
    folder_name='saved_data', timestamp=False
)

save_folder = Path(r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior\saved_data')

tracking = {}


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

    for i, crossing in track(crossings.iterrows(), total=len(crossings)):
        save_name = f'crossing_{ROI}_{i}.json'
        if (save_folder / save_name).exists():
            continue

        if crossing['name'] not in tracking.keys():
                tracking[crossing['name']] = Tracking.get_session_tracking(crossing['name'], movement=False, body_only=False)
                time.sleep(1)

        crossing_trk = ROICrossing.get_crossing_tracking(crossing, tracking[crossing['name']])
        roi_crossing_dict = {**crossing.to_dict(), **crossing_trk}
        
        for k,v in roi_crossing_dict.items():
            if isinstance(v, np.ndarray):
                roi_crossing_dict[k] = list(v)
        
        recorder.add_data(roi_crossing_dict, f'crossing_{ROI}_{i}', fmt="json")

    recorder.add_data(crossings, f"{ROI}_crossings", fmt="h5")


# ---------------------------------------------------------------------------- #
#                             save locomotor bouts                             #
# ---------------------------------------------------------------------------- #
# logger.info('Fetching bouts')
# bouts = pd.DataFrame(
#     (
#         LocomotionBouts
#         & 'complete="true"'
#         & 'direction="outbound"'
#     ).fetch()
# )
# logger.info(f'Got {len(bouts)} bouts')


# bouts_dicts = []
# for i, bout in track(bouts.iterrows(), total=len(bouts)):
#     save_name = f'{i}_complete_bout.json'
#     if (save_folder / save_name).exists():
#         continue
#     else:
#         if bout['name'] not in tracking.keys():
#                 tracking[bout['name']] = Tracking.get_session_tracking(bout['name'], movement=False, body_only=False)
#                 time.sleep(1)
                
#         trk = LocomotionBouts.get_bout_tracking(bout, tracking[bout['name']])
#         trk = {bp:np.array(list(v.values())) for bp, v in trk.to_dict().items()}

#         bout_dict = {**bout.to_dict(), **trk}
#         for k,v in bout_dict.items():
#             if isinstance(v, np.ndarray):
#                 bout_dict[k] = list(v)

    
#         recorder.add_data(bout_dict, f'{i}_complete_bout', fmt="json")
# recorder.describe()
