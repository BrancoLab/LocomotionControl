from vedo.colors import colorMap
import numpy as np
import matplotlib.pyplot as plt
from vedo import Plotter, interactive, Text2D

from proj.rnn.dataset import make_batch

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d
from pyrnn._utils import npify

rnn = RNN.load("task_rnn.pt", n_units=128, input_size=5, output_size=2)

n_frames = 350
n_trials = 25


actors = []
f, axarr = plt.subplots(ncols=2, figsize=(16, 9))

X, Y = make_batch(n_trials)

X = X[:, :n_frames, :]
o, h = rnn.predict_with_history(X)

x = npify(X)

# make vedo plotter
plt = Plotter(shape=(2, 3), size="full", title="my data")

vars = [
    (0, "x pos"),
    (1, "y pos"),
    (2, "theta"),
    (3, "speed"),
    (4, "omega"),
    (5, "time"),
]

# TODO add color by trial side

for n, (idx, name) in enumerate(vars):
    colors = []
    for trialn in range(n_trials):
        if name != "time":
            colors.append(
                colorMap(x[trialn, :, idx], name="plasma", vmin=-1, vmax=1)
            )
        else:
            colors.append(
                colorMap(
                    np.arange(n_frames), name="plasma", vmin=0, vmax=n_frames
                )
            )

    _, actors = render_state_history_pca_3d(
        h,
        alpha=1,
        actors=actors,
        mark_start=True,
        _show=False,
        color=colors,
        lw=0.1,
        color_by_trial=True,
    )

    plt.show(actors, Text2D(name, pos=8), at=n)

print("Render ready")
interactive()

# # Get colors and plot
# colors=[]
# for i in range(X.shape[0]):
#     # if x[i, 0, 3].min() < -.50:
#     #     colors.append('salmon')
#     # else:
#     #     colors.append('skyblue')
#     colors.append(colorMap(x[i, :, 3], name='plasma', vmin=-1, vmax=1))

# #     # Plot data
# #     axarr[0].plot(x[i, :, 0], x[i, :, 1], color=colors[-1][0])
# #     axarr[1].plot(x[i, :, 2], label='$\\theta$', color=deep_purple_light)
# #     axarr[1].plot(x[i, :, 3], label='$v$', color='orange')
# #     axarr[1].plot(x[i, :, 4], label='$\\omega$', color='seagreen')
# #     if i == 0:
# #         axarr[1].legend()
# # plt.show()
