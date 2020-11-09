import matplotlib.pyplot as plt
import os
from vedo import show
from rich.progress import track
import numpy as np
from vedo.colors import colorMap


from pyrnn import CustomRNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn._utils import npify
from proj.rnn.dataset import (
    DeltaStateDataSet,
    plot_predictions,
    make_batch,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

name = None
batch_size = 64
epochs = 100
lr_milestones = [100, 1300]
lr = 0.001
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = DeltaStateDataSet(name=name, dataset_length=1500)

    rnn = CustomRNN.from_json("proj/rnn/rnn.json")

    loss_history = rnn.fit(
        dataset,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lr_milestones=lr_milestones,
        l2norm=0,
        report_path="task_rnn.txt",
        stop_loss=stop_loss,
    )
    rnn.save("task_rnn.pt")

    plot_predictions(rnn, batch_size)
    plot_training_loss(loss_history)
    plt.show()
else:
    rnn = CustomRNN.load(
        "task_rnn.pt", n_units=128, input_size=5, output_size=2
    )

# ------------------------------- Activity PCA ------------------------------- #
actors = []
f, ax = plt.subplots()
colorby = 4
n_frames = 350
for i in track(range(10)):

    X, Y = make_batch()
    x = npify(X)

    if np.min(x[0, :n_frames, 0]) < -0.1:
        c = "r"
    else:
        c = "g"

    ax.plot(x[0, :n_frames, 0], x[0, :n_frames, 1], color=c)

    color = colorMap(X[0, :n_frames, colorby], name="bwr", vmin=-1, vmax=1)

    o, h = rnn.predict_with_history(X[0, :n_frames, :].reshape(1, n_frames, 5))
    _, actors = plot_state_history_pca_3d(
        h,
        alpha=1,
        actors=actors,
        mark_start=True,
        _show=False,
        color=c,
        lw=0.1,
    )
plt.show()
actors = [a.lw(0.001).lc([0.3, 0.3, 0.3]) for a in actors]
print("Ready")
show(*actors)
