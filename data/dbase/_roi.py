import pandas as pd
import numpy as np
from typing import Tuple

from fcutils.progress import track
from fcutils.maths.signals import get_onset_offset

from data.dbase._tracking import calc_angular_velocity
from data import arena
from geometry.vector_utils import smooth_path_vectors
from geometry import Path


def cleanup_path(x:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray]:
    '''
        It time-bins the tangent, velocity and acceleration vectors
        of the path to get cleaner data.
        For the accelration it take the dot product with the tangent vector
        to get the +/- acceleration
    '''
    path = Path(x, y)

    wnd = 5
    velocity, acceleration, tangent = smooth_path_vectors(path, window=wnd)

    return x[wnd:], y[wnd:], velocity.magnitude[wnd:], acceleration.dot(tangent)[wnd:], tangent.angle[wnd:] + 180

def get_rois_crossings(
    tracking: pd.DataFrame,
    ROI: arena.ROI,
    max_duration:float,
    min_speed:float,
) -> dict:
    # get every time the mouse enters the ROI
    enters = sorted([x for x in set(get_onset_offset(tracking.global_coord > ROI.g_0, 0.5)[0]) if x > 0])

    # loop over the results
    results = []

    for start in track(enters, transient=True):
        if tracking.speed[start] < min_speed or start < 60:
            continue

        # check if anywhere the mouse left the ROI
        gcoord = tracking.global_coord[start:start+max_duration+1]
        if gcoord[0] > ROI.g_0 + 0.1:
            continue

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

        if tracking.x.min()<0 or tracking.x.max()>40:
            continue

        duration =  (end - start) / 60
        if duration < .2 or duration > max_duration/60:
            continue

        if tracking.global_coord[end] > ROI.g_1 + 0.05:
            continue
        
        if end <= start:
            continue

        # get cleaner tracking vectors
        x, y, speed, acceleration, theta = cleanup_path(tracking.x[start:end], tracking.y[start:end])
        thetadot = calc_angular_velocity(theta) * 60
        thetadotdot = calc_angular_velocity(thetadot)

        # get data
        results.append(dict(
            roi = ROI.name,
            start_frame = start,
            end_frame = end,
            duration = (end - start) / 60,
            mouse_exits = 1 if exit else -1,
            gcoord = tracking.global_coord[start:end],
            x = x,
            y = y,
            speed = speed,
            theta = theta,
            thetadot = thetadot,
            thetadotdot = thetadotdot,
            acceleration  = acceleration,
        ))

    return results


def select_twin_crossing(crossings:pd.DataFrame, selected_id:int) -> int:
    '''
        Given a selected ROI crossing (specified by its ID) and a dataframe of 
        crossings it selects the fastest crossing with similar initial conditoins.
    '''
    # get the selected crossing
    selected = crossings.iloc[selected_id]
    crossings.drop(selected_id)

    # TODO find the closest twin crossing
    raise NotImplementedError



