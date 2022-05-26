import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append("./")

from pyrnn import RNN

# from pyrnn.render import render_state_history_pca_3d
from analysis.RNN.task import (
    # make_batch,
    GoalDirectedLocomotionDataset,
    plot_predictions,
)

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


SESSION = "RNN_220525_164401"
n_units = 64


save_folder = (
    Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\trained_networks"
    )
    / SESSION
)
dataset = GoalDirectedLocomotionDataset(max_dataset_length=1)


rnn = RNN.load(
    str(save_folder / "MIN_LOSS.pt"),
    n_units=n_units,
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
)

f = plot_predictions(rnn)
plt.ioff()
plt.show()
plt.ion()

# X, Y = make_batch()
# o, h = rnn.predict_with_history(X)


# render_state_history_pca_3d(h, alpha=0.01, lw=0.3)
