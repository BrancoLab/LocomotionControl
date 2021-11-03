# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tpd import recorder
from loguru import logger


from data import paths
from data.data_structures import LocomotionBout
import draw
from data.dbase import db_tables
from geometry import Path
from kinematics import track
from kinematics import track_cordinates_system as TCS

'''
    It plots rasters with spikes of each unit aligned to locomotion onsets and offsets for each session
'''



# %%
# get sessions and tracking
sessions = (db_tables.ValidatedSession * db_tables.Session & 'is_recording=1').fetch('name')
logger.info(f'Found {len(sessions)} recordings')
print(sessions, sep='\n')



# %%

# select a session
session = 'FC_210721_AAA1110750_hairpin'

# ----------------------------- load probe/units ----------------------------- #
logger.info('fetching units')
recording = (db_tables.Recording & f'name="{session}"').fetch(as_dict=True)[0]
cf = recording['recording_probe_configuration']
logger.info("Fetching ephys data")
units = db_tables.Unit.get_session_units(
    session,
    cf,
    spikes=True,
    firing_rate=True,
    frate_window=100,
)
units['probe_configuration'] = [cf] * len(units)
rsites = pd.DataFrame((db_tables.Probe.RecordingSite & recording & f'probe_configuration="{cf}"').fetch())
logger.info(f'Found {len(units)} units')


# %%
# ------------------------------- load tracking ------------------------------ #
tracking_db = db_tables.Tracking.get_session_tracking(session, movement=False)
fast = np.where(tracking_db.speed > 30)[0]

# %%
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = track.extract_track_from_image(
    points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
)

tracking = Path(tracking_db.x[fast], tracking_db.y[fast])
linearized = TCS.path_to_track_coordinates_system(
                center_line, tracking
            )

# %%
from fcutils.maths import derivative

unit = units.iloc[0]


spikes = np.zeros_like(tracking_db.x)
spikes[unit.spikes] = 1

direction = derivative(tracking_db.global_coord)
direction[direction > 0] = 1
direction[direction < 0] = -1

data = pd.DataFrame(dict(x=tracking_db.x, y=tracking_db.y, speed=tracking_db.speed, spikes=spikes, direction=direction))

data = data.loc[(data.speed > 20)&(data.spikes == 1)&(data.direction > 0)]
draw.Tracking.scatter(data.x.values, data.y.values)



# %%

plt.hist(data.speed)