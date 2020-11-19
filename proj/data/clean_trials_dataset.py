# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.prompt import Confirm
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d


"""
    Take raw tracking from the psychometric mazes and clean it up 
    to prepare a dataset for control/rnn work

    Steps:
        1) discard bad traces
        2) smooth traces
        3) augment dataset
"""

# %%

# Load dataset
trials = pd.read_hdf(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials.h5",
    key="hdf",
)
print(f"Loaded {len(trials)} trials")


def rolling_mean(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(
        a1.rolling(window=n, min_periods=1, center=True).mean()
    )

    return moving_avg


# %%
# Manual curation to discard bad trials + smoothing
to_keep = dict(x=[], y=[], orientation=[], speed=[], fps=[],)
for i, t in trials.iterrows():
    # if i > 10:
    #     break

    if t.fps == 30:
        wnd = 8
    else:
        wnd = 10

    x = rolling_mean(t.body_xy[:, 0], wnd)
    y = rolling_mean(t.body_xy[:, 1], wnd)

    ori = rolling_mean(t.body_orientation, wnd)
    speed = rolling_mean(t.body_speed, wnd)
    # Remove those that end short
    if y[-1] < 600 or y[0] > 200:
        continue
    # Select the start to when they're on their way up
    start = np.where(y > 200)[0][0]
    x, y, ori, speed = x[start:], y[start:], ori[start:], speed[start:]

    # plot and choose
    f, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, color="k", lw=1)
    plt.show()

    if Confirm.ask(f"Keep {i} of {len(trials)}?"):
        to_keep["x"].append(x)
        to_keep["y"].append(y)
        to_keep["orientation"].append(ori)
        to_keep["speed"].append(speed)
        to_keep["fps"].append(t.fps)

print(f'Keeping {len(to_keep["x"])} trials')
pd.DataFrame(to_keep).to_hdf(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials_cleaned.h5",
    key="hdf",
)

# %%
# Plot cleaned trials
trials = pd.read_hdf(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials_cleaned.h5",
    key="hdf",
)

# ! This also recomputes orientation from the x,y tracking which should've happened before

# Augment trials by flipping on X axis
augmented = dict(x=[], y=[], orientation=[], speed=[], fps=[],)

f, ax = plt.subplots(figsize=(16, 16))
for i, t in trials.iterrows():
    # Fix real trial
    ori = calc_angle_between_points_of_vector_2d(t.x, t.y)
    ori[0] = ori[1]

    augmented["x"].append(t.x)
    augmented["y"].append(t.y)
    augmented["orientation"].append(ori)
    augmented["speed"].append(t.speed)
    augmented["fps"].append(t.fps)

    # Flip on the X axis
    augmented["x"].append(1000 - t.x)
    augmented["y"].append(t.y)
    augmented["orientation"].append(360 - ori)
    augmented["speed"].append(t.speed)
    augmented["fps"].append(t.fps)
pd.DataFrame(augmented).to_hdf(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials_augmented.h5",
    key="hdf",
)


# Plot all trials
trials = pd.read_hdf(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials_augmented.h5",
    key="hdf",
)
print(f"Keeping {len(trials)} trials")

f, ax = plt.subplots(figsize=(16, 16))
for i, t in trials.iterrows():
    ax.plot(t.x, t.y, color="k", lw=1)

# %%
