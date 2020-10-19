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

# ! TODO look into time steps resolution and match it to simulations
# TODO ! https://stackoverflow.com/questions/54009661/what-is-the-timestep-in-keras-lstm

# TODO look into normalizations etc.
# TODO parameters grid search
