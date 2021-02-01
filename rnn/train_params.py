import sys

sys.path.append("./")
from rnn.dataset import datasets

"""
    params for training an RNN on the control task.
    These are used by rnn.train.py to carry out the training, 
    they're kept here for easier editing via terminal
    when training on winstor
"""
# ---------------------------------- Dataset --------------------------------- #
N_trials = -1  # number of trials to use, set to -1 to use entire dataset

dataset_name = "dataset_predict_" + "tau_from_deltaXYT"

try:
    DATASET = datasets[dataset_name]
except KeyError:
    raise KeyError(
        f"Could not find dataset {dataset_name}, available dataset: {datasets.keys()}"
    )

name = "" + "_" + DATASET.name  # rnn name

# ------------------------------------ RNN ----------------------------------- #
n_units = 256  # number of units in the RNN
dale_ratio = None
autopses = True
w_in_bias = False
w_in_train = True
w_out_bias = False
w_out_train = True
l2norm = 0  # recurrent weights normalization


# ---------------------------------- dataset --------------------------------- #

# set some variables controlling datasets generation
DATASET.augment_probability = (
    0.0  # probabily  of augmenting a trial during training
)

DATASET.to_chunks = (
    False  # if true trials are cut into chunks of given lengths
)
DATASET.chunk_length = 64

DATASET.warmup = False  # add a warmup phase to start of trials
DATASET.warmup_len = 64

DATASET.smoothing_window = 61  # used to smooth inputs and outputs

# --------------------------------- training --------------------------------- #

batch_size = 2048
epochs = 25000
lr_milestones = [1000, 6000, 15000]
lr = 0.001
stop_loss = None
