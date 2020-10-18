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

# TODO think about how to improve dataset creation for real trajectory as input
# TODO make simulator work
