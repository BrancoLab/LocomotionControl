from proj.rnn.train import RNNTrainer

# def train_generator():
#     while True:
#         x, y, mask, trial_params = task.get_trial_batch()
#         yield x, y

trainer = RNNTrainer()

trainer.train()

# TODO add 'log_loss_every' key for custom traceback class
# TODO fill in and save a copy of config in RNN folder
# TODO save a copy of log to file
