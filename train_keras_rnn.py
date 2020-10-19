from proj.rnn.dataset import DatasetMaker
from proj.rnn.train import RNNTrainer

train = True

# ? Make dataset
if not train:
    maker = DatasetMaker()
    maker.make_dataset()

# ? Train
if train:
    trainer = RNNTrainer()
    trainer.train()

# TODO look into normalizations etc.
# TODO parameters grid search
# TODO set it up to work on HPC

"""
C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\libnvvp;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;
"""
