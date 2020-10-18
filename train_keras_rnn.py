from proj.rnn.dataset import DatasetMaker
from proj.rnn.train import RNNTrainer

# ? Make dataset
maker = DatasetMaker()
maker.make_dataset()


# ? Train
trainer = RNNTrainer()
trainer.train()
