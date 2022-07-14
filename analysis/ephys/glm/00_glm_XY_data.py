import sys
from fcutils.maths import derivative
from fcutils.maths.signals import rolling_mean
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from scipy import interpolate
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")


from analysis.ephys.utils import (
    get_recording_names,
    get_data,
    get_session_bouts,
    trim_bouts,
)

save_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys")
cache = Path(r"D:\GLM\data")

REGION = "CUN/PPN"
recordings = get_recording_names(region=REGION)
recordings

# ---------------------------------- params ---------------------------------- #
curvature_horizon = 30
curvature_sampling_spacing = 5
curv_sample_points = np.arange(
    0,
    curvature_horizon + curvature_sampling_spacing,
    curvature_sampling_spacing,
)
minimum_bout_ds = 100
track_downsample_factor = 25
firing_rate_gaussian = 250  # width in ms


# Process data
def upsample_frames_to_ms(var):
    """
        Interpolates the values of a variable expressed in frams (60 fps)
        to values expressed in milliseconds.
    """
    t_60fps = np.arange(len(var)) / 60
    f = interpolate.interp1d(t_60fps, var)

    t_1000fps = np.arange(0, t_60fps[-1], step=1 / 1000)
    # t_200fps = np.arange(0, t_60fps[-1], step=1/200)
    interpolated_variable_values = f(t_1000fps)
    return interpolated_variable_values


# upsample
def load_get_recording_data(REC):
    # load data
    units, left_fl, right_fl, left_hl, right_hl, body = get_data(REC)
    if not len(units):
        return None, None, None, None, None, None, None

    bouts = trim_bouts(get_session_bouts(REC, complete=None, direction=None))
    n = len(bouts)
    bouts["ds"] = [abs(b.s[-1] - b.s[0]) for i, b in bouts.iterrows()]
    bouts = bouts.loc[bouts.ds >= minimum_bout_ds]
    print(f"    loaded {n} bouts, kept {len(bouts)} bouts")

    x = upsample_frames_to_ms(body.x)
    y = upsample_frames_to_ms(body.y)

    return bouts, x, y


def dorec(REC):
    if (cache / f"{REC}_bouts_xy.h5").exists():
        print(f"{REC}_bouts_xy.h5 already exists")
        return
    print(f"Processing {REC}")
    bouts, x, y = load_get_recording_data(REC)

    # for i, bout in track(bouts.iterrows(), total=len(bouts), description=REC):
    for i, bout in bouts.iterrows():
        bout_savepath = cache / f"{REC}_bout_{bout.start_frame}_xy.feather"
        if bout_savepath.exists():
            print(f"{REC}_bout_{bout.start_frame}.feather already exists")
            continue

        S = upsample_frames_to_ms(bout.s)
        start_ms = int(bout.start_frame / 60 * 1000)
        end_ms = start_ms + len(S)
        data = dict(
            x=x[start_ms:end_ms],
            y=y[start_ms:end_ms],
            s=S,
            ds=derivative(rolling_mean(S, 101)) * 1000,
        )

        pd.DataFrame(data).to_feather(bout_savepath)
        del data


def collate_rec(REC):
    savepath = cache / f"{REC}_bouts_xy.h5"
    if savepath.exists():
        print(f"{REC}_bouts_xy.h5 already exists")
        return

    print(f"Collating {REC}")

    bouts = trim_bouts(get_session_bouts(REC, complete=None, direction=None))
    bouts["ds"] = [abs(b.s[-1] - b.s[0]) for i, b in bouts.iterrows()]
    bouts = bouts.loc[bouts.ds >= minimum_bout_ds]

    bouts_files = list(cache.glob(f"{REC}_bout_*_xy.feather"))

    if len(bouts_files) < len(bouts):
        print(
            f"    Not all bouts were saved for {REC}: {len(bouts_files)}/{len(bouts)}"
        )
        print(*bouts_files, sep="\n")
        return

    bouts_data = []
    for i, bout in bouts.iterrows():
        try:
            f = cache / f"{REC}_bout_{bout.start_frame}_xy.feather"
            _data = pd.read_feather(f)
        except:
            logger.warning(f"Failed to read file for bout {i}: {f}")
            return
        bouts_data.append(_data.astype(np.float32))

    if not len(bouts_data):
        print(f"    No bouts were saved for {REC}")
        return
    bouts_data = pd.concat(bouts_data)

    print(f" Saving data (shape: {bouts_data.shape})")
    bouts_data.to_hdf(savepath, key="data")
    print(" Saved all data")


if __name__ == "__main__":
    from multiprocessing import Pool

    pool = Pool(4)

    recordings = get_recording_names(region=REGION)
    pool.map(dorec, recordings)
    pool.map(collate_rec, recordings)

    # for rec in get_recording_names(region=REGION):
    #     dorec(rec)
