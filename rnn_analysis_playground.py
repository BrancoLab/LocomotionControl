from pathlib import Path
from pyrnn.rnn import RNN
import matplotlib.pyplot as plt
import torch

from rnn.dataset import plot_predictions
from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET


# ----------------------------------- load ----------------------------------- #

fld = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Apps/loco_upload/201215_091914_RNN_vanilla_large_batch_small_lr_dataset_predict_nudot_from_xyt/"
)
rnn = RNN.load(
    str([f for f in fld.glob("*.pt")][0]),
    n_units=256,
    input_size=3,
    output_size=2,
    on_gpu=False,
    load_kwargs=dict(map_location=torch.device("cpu")),
)
data = DATASET()


# ----------------------------------- plot ----------------------------------- #

for i in range(10):
    plot_predictions(rnn, data)
plt.show()

# TODO asses quality somehow
# TODO get hidden state
# TODO RNN hidden state plots
