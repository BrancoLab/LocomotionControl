import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import numpy as np
from typing import List

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout

'''
    Take complete, nice bouts and extract for each frame:
        - distance in polar coord after T ms
        - orientation after T ms
        - current speed
        - current angular velocity
'''

def load_bouts(min_dur:float=10, keep:int=-1) -> List[LocomotionBout]:
    _bouts = pd.read_hdf(
        paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
    ).sort_values("duration")
    _bouts = _bouts.loc[_bouts.duration < min_dur]
    
    _bouts = _bouts.iloc[:keep]

    print(f"Kept {len(_bouts)} bouts")

    bouts = []
    for i, bout in _bouts.iterrows():
        bouts.append(LocomotionBout(bout))
    return bouts


def get_IO_from_bout(bout:LocomotionBout, dt:float=250, speeds_dt:float=60) -> dict:
    output = dict(
        rho=[],  # distance
        phi=[],  # angle
        theta=[], # orientation
        speed=[],
        avel=[],
        target_speed=[],  # speed/avel at next frame
        target_avel=[],
    )

    # future time point distance
    dt = int(dt * 60 / 1000) # ms to frames
    if dt < 2:
        raise ValueError('dt too small')
    elif dt > 20:
        raise ValueError('dt too large')
    speeds_dt = int(speeds_dt * 60 / 1000)
    if speeds_dt < 1:
        raise ValueError('speeds_dt too small')

    for t in range(len(bout)-dt):
        t0 = bout.at(t)
        tnext = bout.at(t+speeds_dt)
        t1 = bout.at(t+dt)

        # get distance to target state in polar coordiantes
        vec = t1.xy - t0.xy
        output['rho'].append(vec.magnitude)
        output['phi'].append(vec.angle)

        # get target orientation
        output['theta'].append(t1.theta)

        # get current speeds
        output['speed'].append(t0.s)
        output['avel'].append(t0.thetadot)

        # get next speeds
        output['target_speed'].append(tnext.s)
        output['target_avel'].append(tnext.thetadot)

    output = {k:np.array(v) for k,v in output.items()}
    return output

if __name__ == '__main__':
    bouts = load_bouts(keep=2)
    IO = get_IO_from_bout(bouts[0])

    import matplotlib.pyplot as plt

    plt.plot(IO['target_speed']-IO['speed'])
    plt.show()