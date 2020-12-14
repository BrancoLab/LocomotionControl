from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET

"""
    params for training an RNN on the control task.
    These are used by rnn.train.py to carry out the training, 
    they're kept here for easier editing via terminal
    when training on winstor
"""

# ---------------------------------- Dataset --------------------------------- #
N_trials = 10  # number of trials to use, set to -1 to use entire dataset
name = "" + "_" + DATASET.name

# ------------------------------------ RNN ----------------------------------- #
n_units = 256  # number of units in the RNN
dale_ratio = None
autopses = True
w_in_bias = False
w_in_train = False
w_out_bias = False
w_out_train = False


# --------------------------------- Training --------------------------------- #
batch_size = 64
epochs = 5  # 300
lr_milestones = [500, 4000]
lr = 0.001
stop_loss = None


if __name__ == "__main__":
    # Create a dataset from the raw data
    DATASET().make()
    DATASET().plot_random()
    DATASET().plot_durations()
