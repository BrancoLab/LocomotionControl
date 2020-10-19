from proj.rnn.dataset import DatasetMaker
from proj.rnn.train import RNNTrainer

train = False

# ? Make dataset
if not train:
    maker = DatasetMaker()
    maker.make_dataset()


# ? Train
if train:
    trainer = RNNTrainer()
    trainer.train()

# TODO copy dataset to saved RNN folder?
# TODO look into normalizations etc.
# TODO parameters grid search
# TODO set it up to work on HPC and use GPU
# TODO train/test split
