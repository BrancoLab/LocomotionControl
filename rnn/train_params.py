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
    PredictNudotFromDeltaXYT=PredictNudotFromDeltaXYT,
    PredictTauFromDeltaXY=PredictTauFromDeltaXYT,
)

"""
    params for training an RNN on the control task.
    These are used by rnn.train.py to carry out the training, 
    they're kept here for easier editing via terminal
    when training on winstor
"""

# ---------------------------------- Dataset --------------------------------- #
N_trials = -1  # number of trials to use, set to -1 to use entire dataset

dataset_name = "PredictNuDotFromXYT"
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
DATASET.augment_probability = (
    0.0  # probabily  of augmenting a trial during training
)

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
