import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import interpolate
from loguru import logger
from fcutils.maths.signals import rolling_mean
import warnings
from scipy.ndimage.filters import gaussian_filter1d


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

REGION = "MOs"


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
firing_rate_gaussian = 100  # width in ms


def get_track_data():
    track_data = pd.read_json(
        r"C:\Users\Federico\Documents\GitHub\pysical_locomotion\analysis\ephys\track.json"
    ).iloc[::track_downsample_factor]
    track_data = track_data.reset_index(drop=True)
    S_f = track_data.S.values[-1]
    track_data

    # load track from json
    k_shifts = np.arange(curvature_horizon + 1)
    curv_shifted = {
        **{f"k_{k}": [] for k in k_shifts},
        **{f"idx_{k}": [] for k in k_shifts},
    }
    for i, s in enumerate(track_data.S):
        for k in k_shifts:
            if s + k < S_f:
                select = track_data.loc[track_data.S >= s + k]
                curv_shifted[f"idx_{k}"].append(select.index[0])
                curv_shifted[f"k_{k}"].append(select["curvature"].iloc[0])
            else:
                curv_shifted[f"k_{k}"].append(np.nan)
                curv_shifted[f"idx_{k}"].append(np.nan)

        # break

    for k, v in curv_shifted.items():
        track_data.insert(2, k, rolling_mean(v, 201))
    track_data.head()

    # get distance from the next curve apex based on the direction of travel

    from scipy.signal import find_peaks

    k = rolling_mean(np.abs(track_data.curvature.values), 251)
    peaks, _ = find_peaks(k, prominence=0.02)

    # peaks_idxs, _ = find_peaks(k, prominence=0.02)
    peaks = track_data.iloc[peaks]
    peaks

    apex_distance = dict(outward=[], invard=[],)

    for i, row in track_data.iterrows():
        # find next peak with larger s
        next_peak = peaks[peaks.S > row.S]
        if len(next_peak) > 0:
            # get distance from it
            next_peak = next_peak.iloc[0]
            apex_distance["outward"].append(next_peak.S - row.S)
        else:
            apex_distance["outward"].append(np.nan)

        # get distance from previous peak
        prev_peak = peaks[peaks.S < row.S]
        if len(prev_peak) > 0:
            prev_peak = prev_peak.iloc[-1]
            apex_distance["invard"].append(row.S - prev_peak.S)
        else:
            apex_distance["invard"].append(np.nan)

    track_data["apex_distance_outward"] = apex_distance["outward"]
    track_data["apex_distance_inward"] = apex_distance["invard"]
    return track_data


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


def calc_firing_rate(spikes_train: np.ndarray, dt: int = 10):
    """
        Computes the firing rate given a spikes train (wether there is a spike or not at each ms).
        Using a gaussian kernel with standard deviation = dt/2 [dt is in ms]
    """
    return gaussian_filter1d(spikes_train, dt) * 1000


# upsample
def load_get_recording_data(REC):
    # load data
    units, left_fl, right_fl, left_hl, right_hl, body = get_data(REC)
    if not len(units):
        logger.warning(f"No units found for {REC}")
        return None, None, None, None, None, None, None

    bouts = trim_bouts(
        get_session_bouts(REC, complete="true", direction="outbound")
    )
    n = len(bouts)
    bouts["ds"] = [abs(b.s[-1] - b.s[0]) for i, b in bouts.iterrows()]
    bouts = bouts.loc[bouts.ds >= minimum_bout_ds]
    print(f"    loaded {n} bouts, kept {len(bouts)} bouts")

    v = upsample_frames_to_ms(body.speed)
    omega = upsample_frames_to_ms(body.thetadot)

    dv_300ms = np.hstack([v[300:], v[300] * np.ones(300)]) - v
    domega_300ms = np.hstack([omega[300:], omega[300] * np.ones(300)]) - omega

    # get unit firing rate in milliseconds
    if REGION == "MOs":
        units = units.loc[
            units.brain_region.isin(
                ["MOs", "MOs1", "MOs2/3", "MOs5", "MOs6a", "MOs6b"]
            )
        ]
    else:
        units = units.loc[units.brain_region.isin(["CUN", "PPN"])]

    # go over each unt, get firing rate and make shuffles and save
    units_names = []
    for i, unit in units.iterrows():
        name = f"{REC}_{unit.unit_id}.npy"
        unit_save = Path(f"D:\\GLM\\tmp\\{name}")
        units_names.append(name)

        # get firing rate
        if not unit_save.exists():
            time = np.zeros(len(v))  # time in milliseconds
            spikes_times = np.int64(np.round(unit.spikes_ms))
            spikes_times = spikes_times[spikes_times < len(time)]
            time[spikes_times] = 1

            fr = calc_firing_rate(
                time, dt=firing_rate_gaussian
            )  # firing rate at 1000 fps
            np.save(unit_save, fr)
        else:
            fr = np.load(unit_save)

        # make suffles
        N = 100
        for n in range(N):
            name = f"{REC}_{unit.unit_id}_shuffle_{n}.npy"
            unit_save = Path(f"D:\\GLM\\tmp\\{name}")
            units_names.append(name)

            if not unit_save.exists():
                shuffle = np.random.randint(
                    10 * 1000, 100 * 1000
                )  # shuffle between 10 and 100 seconds

                fr_shuffled = np.hstack([fr[shuffle:], fr[:shuffle]])
                np.save(unit_save, fr_shuffled)

    return units_names, body, bouts, v, omega, dv_300ms, domega_300ms


# Collect data for all bouts


def dorec(REC):
    if (cache / f"{REC}_bouts.h5").exists():
        print(f"{REC}_bouts.h5 already exists")
        return
    print(f"Processing {REC}")
    track_data = get_track_data()
    (
        units_names,
        body,
        bouts,
        v,
        omega,
        dv_300ms,
        domega_300ms,
    ) = load_get_recording_data(REC)

    if units_names is None:
        return

    # units_data = {unit: np.load(f"D:\\GLM\\tmp\\{unit}") for unit in units_names}

    # for i, bout in track(bouts.iterrows(), total=len(bouts), description=REC):
    for i, bout in bouts.iterrows():
        bout_savepath = cache / f"{REC}_bout_{bout.start_frame}.feather"
        if bout_savepath.exists():
            # print(f"{REC}_bout_{bout.start_frame}.feather already exists")
            continue

        S = upsample_frames_to_ms(bout.s)
        start_ms = int(bout.start_frame / 60 * 1000)
        end_ms = start_ms + len(S)
        # n_samples = end_ms - start_ms

        data = dict(apex_distance=[])
        # data['s'].extend(S)

        # get distance from apex
        dist = (
            track_data["apex_distance_outward"]
            if bout.direction == "outward"
            else track_data["apex_distance_inward"]
        )
        for _s, s in enumerate(S):
            idx = np.argmin((track_data.S - s) ** 2)
            data["apex_distance"].append(dist[idx])

        data["v"] = v[start_ms:end_ms]
        data["dv_300ms"] = dv_300ms[start_ms:end_ms]
        data["omega"] = omega[start_ms:end_ms]
        data["domega_300ms"] = domega_300ms[start_ms:end_ms]

        # get firing rate
        for i, unit in enumerate(units_names):
            fr = np.load(f"D:\\GLM\\tmp\\{unit}")
            # fr = units_data[unit]
            unit_id = unit.split("hairpin_")[-1][:-4]
            data[unit_id] = fr[start_ms:end_ms].astype(np.float32)

        # ensure all entries have the same number of samples
        lengths = set([len(v) for v in data.values()])
        if len(lengths) > 1:
            lns = {k: len(v) for k, v in data.items()}
            raise ValueError(f"Lengths of data are not the same:\n{lns}")

        # pd.DataFrame(data).to_hdf(bout_savepath, key="data")
        pd.DataFrame(data).to_feather(bout_savepath)
        del data


if __name__ == "__main__":
    from multiprocessing import Pool

    pool = Pool(6)

    recordings = get_recording_names(region=REGION)
    pool.map(dorec, recordings)

    # for rec in get_recording_names(region=REGION):
    #     dorec(rec)
