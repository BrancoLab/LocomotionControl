import pandas as pd
import numpy as np

from fcutils.maths import derivative
from fcutils.maths.signals import get_onset_offset

from data import arena

def get_rois_crossings(
    tracking: pd.DataFrame,
    ROI: arena.ROI,
    max_duration:float,
) -> dict:
    # get every time the mouse enters the ROI
    enters = sorted([x for x in set(get_onset_offset(tracking.global_coord > ROI.g_0, 0.5)[0]) if x > 0])

    # loop over the results
    results = []

    for start in enters:
        # check if anywhere the mouse left the ROI
        gcoord = tracking.global_coord[start:start+max_duration+1]

        if np.any(gcoord < ROI.g_0):
            continue
        elif np.any(gcoord > ROI.g_1):
            end = np.where(tracking.global_coord[start:] >= ROI.g_1)[0][0] + start
            exit = True
        else:
            end = start + max_duration
            exit = False

        if end > len(tracking.x):
            continue

        duration =  (end - start) / 60
        if duration < .2 or duration > max_duration/60:
            continue

        if tracking.global_coord[end] > ROI.g_1 + 0.05:
            continue
        
        if end <= start:
            continue

        # get data
        results.append(dict(
            roi = ROI.name,
            start_frame = start,
            end_frame = end,
            duration = (end - start) / 60,
            mouse_exits = 1 if exit else -1,
            gcoord = tracking.global_coord[start:end],
            x = tracking.x[start:end],
            y = tracking.y[start:end],
            speed = tracking.speed[start:end],
            direction_of_movement = tracking.direction_of_movement[start:end],
        ))

    return results