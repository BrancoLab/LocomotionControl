import sys
from pathlib import Path
from tpd import recorder
import time
import pandas as pd
from loguru import logger
from fcutils.maths import derivative
sys.path.append("./")


from data.dbase.db_tables import ROICrossing, LocomotionBouts, Tracking
from data import arena






# # save roi crossings
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
    recorder.add_data(crossings, f"{ROI}_crossings", fmt="h5")

# save locomotor bouts
bouts = pd.DataFrame(
    (
        LocomotionBouts
        & 'complete="true"'
        & 'direction="outbound"'
    ).fetch()
)

# add tracking
sessions = bouts["name"].unique()
tracking = {}
for sess in sessions:
    tracking[sess] = Tracking.get_session_tracking(
        sess, body_only=True, movement=False
    )
    time.sleep(.5)

to_add = dict(
    x=[],
    y=[],
    speed=[],
    acceleration=[],
    theta=[],
    thetadot=[],
    thetadotdot=[],
    gcoord=[]
)
for i, bout in bouts.iterrows():
    trk = tracking[bout["name"]]
    
    for k in to_add.keys():
        if k != 'gcoord':
            to_add[k].append(trk[k][bout.start_frame : bout.end_frame].copy())
    to_add['gcoord'].append(trk["global_coord"][bout.start_frame : bout.end_frame].copy())

bouts = pd.concat([bouts, pd.DataFrame(to_add)], axis=1)
logger.info(f"Found {len(bouts)} complete bouts")
recorder.add_data(bouts, f"complete_bouts", fmt="h5")

recorder.describe()
