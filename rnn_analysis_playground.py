from pathlib import Path
from pyrnn.rnn import RNN
import matplotlib.pyplot as plt

from rnn.dataset import plot_predictions
from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET


# ----------------------------------- load ----------------------------------- #

fld = Path(
    "D:\\Dropbox (UCL)\\Apps\\loco_upload\\201211_144704_RNN_dataset_predict_nudot_from_xyt_best"
)
rnn = RNN.load(
    str([f for f in fld.glob("*.pt")][0]),
    n_units=256,
    input_size=3,
    output_size=2,
)
data = DATASET()


# ----------------------------------- plot ----------------------------------- #

for i in range(10):
    plot_predictions(rnn, 100, data)
plt.show()

# TODO make plots plot different trials
# TODO asses quality somehow
# TODO get hidden state
# TODO RNN hidden state plots
