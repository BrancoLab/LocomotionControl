import os

import sys
from pathlib import Path

sys.path.append("./")

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d
from analysis.RNN.task import make_batch, GoalDirectedLocomotionDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
save_folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/trained_networks"
)

dataset = GoalDirectedLocomotionDataset(max_dataset_length=1)
n_units = 64
rnn = RNN.load(
    save_folder / "gld.pt",
    n_units=n_units,
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
)

X, Y = make_batch()
o, h = rnn.predict_with_history(X)

render_state_history_pca_3d(h, alpha=0.01, lw=0.3)
