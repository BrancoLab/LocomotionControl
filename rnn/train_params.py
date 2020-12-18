from rnn.dataset.dataset import (
    PredictTauFromXYT,
    PredictNuDotFromXYT,
    PredictTauFromXYTVO,
    PredictNudotFromDeltaXYT,
    PredictTauFromDeltaXYT,
)

datasets_dict = dict(
    PredictTauFromXYT=PredictTauFromXYT,
    PredictNuDotFromXYT=PredictNuDotFromXYT,
    PredictTauFromXYTVO=PredictTauFromXYTVO,
    PredictNuDotFromDeltaXYT=PredictNudotFromDeltaXYT,
    PredictTauFromDeltaXYT=PredictTauFromDeltaXYT,
)

"""
    params for training an RNN on the control task.
    These are used by rnn.train.py to carry out the training, 
    they're kept here for easier editing via terminal
    when training on winstor
"""
# ---------------------------------- Dataset --------------------------------- #
N_trials = -1  # number of trials to use, set to -1 to use entire dataset

dataset_name = "PredictTauFromXYT"
DATASET = datasets_dict[dataset_name]

name = "" + "_" + DATASET.name  # rnn name

# ------------------------------------ RNN ----------------------------------- #
n_units = 256  # number of units in the RNN
dale_ratio = None
autopses = True
w_in_bias = False
w_in_train = False
w_out_bias = False
w_out_train = False


# --------------------------------- Training --------------------------------- #

# set some variables controlling datasets generation
DATASET.augment_probability = (
    0.0  # probabily  of augmenting a trial during training
)

DATASET.to_chunks = (
    False  # if true trials are cut into chunks of given lengths
)
DATASET.chunk_length = 64

DATASET.warmup = True  # add a warmup phase to start of trials

batch_size = 1024
epochs = 10000
lr_milestones = [500, 4000, 8000]
lr = 0.001
stop_loss = None


if __name__ == "__main__":
    # Create a dataset from the raw data
    DATASET().make()
    DATASET().plot_random()
    DATASET().plot_durations()
