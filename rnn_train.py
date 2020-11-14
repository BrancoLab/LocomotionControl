import matplotlib.pyplot as plt
import os


from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from proj.rnn.dataset import (
    DeltaStateDataSet,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
name = None
batch_size = 64
epochs = 100
lr_milestones = [100, 1300]
lr = 0.001
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #
dataset = DeltaStateDataSet(name=name, dataset_length=1500)

rnn = RNN.from_json("proj/rnn/rnn.json")

loss_history = rnn.fit(
    dataset,
    n_epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    lr_milestones=lr_milestones,
    l2norm=0,
    report_path="task_rnn.txt",
    stop_loss=stop_loss,
)
rnn.save("task_rnn.pt")

plot_predictions(rnn, batch_size)
plot_training_loss(loss_history)
plt.show()
