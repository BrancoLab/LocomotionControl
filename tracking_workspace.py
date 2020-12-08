# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
    blue_grey_darker,
)
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from fcutils.maths.utils import rolling_mean
from scipy.signal import find_peaks


folder = Path(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\Zane"
)
# %%
turners = [
    "ZM_201002_ZM012_escconcat_0.h5",
    "ZM_201002_ZM011_escconcat_5.h5",
    "ZM_201002_ZM012_escconcat_0.h5",
    "ZM_201002_ZM012_escconcat_6.h5",
    "ZM_201002_ZM012_escconcat_8.h5",
    "ZM_201002_ZM012_escconcat_10.h5",
    "ZM_201002_ZM012_escconcat_12.h5",
    "ZM_201002_ZM012_escconcat_15.h5",
    "ZM_201002_ZM012_escconcat_6.h5",
    "ZM_201002_ZM014_escconcat_14.h5",
    "ZM_201002_ZM015_escconcat_4.h5",
    "ZM_201002_ZM015_escconcat_8.h5",
    "ZM_201002_ZM015_escconcat_12.h5",
    "ZM_201003_ZM017_escconcat_11.h5",
    "ZM_201003_ZM018_escconcat_0.h5",
    "ZM_201003_ZM018_escconcat_6.h5",
    "ZM_201003_ZM018_escconcat_8.h5",
    "ZM_201003_ZM018_escconcat_9.h5",
    "ZM_201003_ZM020_escconcat_2.h5",
    "ZM_201003_ZM020_escconcat_5.h5",
    "ZM_201003_ZM020_escconcat_8.h5",
    "ZM_201003_ZM020_escconcat_19.h5",
    "ZM_201003_ZM020_escconcat_23.h5",
]


files = [f for f in folder.glob("*.h5") if f.name in turners]
every = 15

paws = ("left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw")
paw_colors = {
    "left_forepaw": indigo_darker,
    "right_forepaw": salmon_darker,
    "left_hindpaw": salmon,
    "right_hindpaw": indigo,
}

cm_per_px = 1 / 30.8
# %%


def line(bp1, bp2, ax, **kwargs):
    x1 = tracking[f"{bp1}_x"].values[frames]
    y1 = tracking[f"{bp1}_y"].values[frames]
    x2 = tracking[f"{bp2}_x"].values[frames]
    y2 = tracking[f"{bp2}_y"].values[frames]

    ax.plot([x1, x2], [y1, y2], **kwargs)


def point(bp, ax, **kwargs):
    x = tracking[f"{bp}_x"].values[frames]
    y = tracking[f"{bp}_y"].values[frames]

    ax.scatter(x, y, **kwargs)


def draw_mouse(ax, **kwargs):
    patches = []
    for n in frames:
        bps = (
            "tail_base",
            "left_hindpaw",
            "left_forepaw",
            "snout",
            "right_forepaw",
            "right_hindpaw",
        )
        x = [tracking[f"{bp}_x"].values[n] for bp in bps]
        y = [tracking[f"{bp}_y"].values[n] for bp in bps]
        patches.append(Polygon(np.vstack([x, y]).T, True, lw=None))

    p = PatchCollection(patches, alpha=0.3, color=blue_grey_darker, lw=None)
    ax.add_collection(p)


def t(d):
    try:
        d = d.values
    except Exception:
        pass

    return rolling_mean(d[start:], 3) * cm_per_px


def get_frames():
    peaks, _ = find_peaks(t(tracking["left_bone_length"]))
    return peaks + start


def r(a):
    return np.degrees(np.unwrap(np.radians(a)))


# %%

starts = [
    71,
    58,
    60,
    55,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


for runn, (f, start) in enumerate(zip(files, starts)):
    if runn != 3:
        continue

    tracking = pd.read_hdf(f, key="hdf")

    f = plt.figure(constrained_layout=True, figsize=(16, 12))
    gs = f.add_gridspec(3, 4)

    tracking_ax = f.add_subplot(gs[:, 0])
    tracking_ax.axis("equal")
    paws_ax = f.add_subplot(gs[0, 1:])
    bones_ax = f.add_subplot(gs[1, 1:])
    diag_ax = f.add_subplot(gs[2, 1:])

    # Plot body
    frames = get_frames()
    draw_mouse(tracking_ax)

    # Plot paws
    for paw, color in paw_colors.items():
        point(paw, tracking_ax, zorder=1, color=color, s=50)

    line(
        "left_hindpaw",
        "right_forepaw",
        tracking_ax,
        color=salmon,
        lw=2,
        zorder=2,
    )
    line(
        "right_hindpaw",
        "left_forepaw",
        tracking_ax,
        color=indigo,
        lw=2,
        zorder=2,
    )

    # Plot paw speeds
    for n, (paw, color) in enumerate(paw_colors.items()):
        y = t(tracking[f"{paw}_speed"])

        paws_ax.plot(y, color=color, lw=3, alpha=0.8, label=paw)
    paws_ax.legend()

    # Plot bone lengths
    bones_ax.plot(
        t(tracking["left_bone_length"]),
        color=salmon_darker,
        lw=3,
        label="LF-LH distance",
    )
    bones_ax.plot(
        t(tracking["right_bone_length"]),
        color=indigo_darker,
        lw=3,
        label="RF-RH distance",
    )
    bones_ax.legend()

    # Plot orientation
    diag_ax.plot(
        t(r(tracking["lower_body_bone_orientation"])),
        label="body angle",
        color=blue_grey_darker,
    )
    diag_ax.legend()

    # mark snapshots
    for n in frames:
        for ax in (paws_ax, bones_ax, diag_ax):
            ax.axvline(n - start, lw=2, color=[0.6, 0.6, 0.6])

    break
# %%

# %%
