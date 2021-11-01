# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# import torch
# import torch.utils.data as data

# from fcutils.path import from_yaml
# from fcutils.progress import track
# from fcutils.maths.coordinates import cart2pol

# """
#     Locomotion task.
#         The RNN receives 2 inputs:
#             - position X  milliseconds in the future
#             - orientation X  milliseconds in the future

#         and has to produce two outputs:
#             - speed
#             - avel

#         In the experiment design both inputs go to MOs while the outputs
#         come out of CUN and GRN respectively.

#         The outputs have to match those of actual mice.
# """

# is_win = sys.platform == "win32"


# class ThreeBitDataset(data.Dataset):
#     """
#     creates a pytorch dataset for loading
#     the data during training.
#     """

#     def __init__(self):
#         self.params = from_yaml('./RNN/params.yaml')

#         self.load_bouts()
#         self.make_trials()

#     def __len__(self):
#         return self.dataset_length

#     def __getitem__(self, item):
#         X_batch, Y_batch = self.items[item]
#         return X_batch, Y_batch

#     def load_bouts(self):
#         '''
#             Loads tracking data from file
#         '''
#         self.bouts = pd.read_hdf(self.params['tracking_data_path'])
#         self.bouts = self.bouts.loc[self.bouts.duration <= self.params['max_bout_duration']]

#         self.dataset_length = len(self.bouts)
#         self.sequence_length = self.params['max_bout_duration'] * self.params['fps']

#     def make_trials(self):
#         """
#         Generate the set of trials to be used fpr traomomg
#         """
#         seq_len = self.sequence_length
#         dset_len = self.dataset_length

#         self.items = {}
#         for i in track(
#             range(dset_len),
#             description="Generating data...",
#             total=dset_len,
#             transient=True,
#         ):

#             # get bout
#             bout = self.bouts.iloc[i]

#             # get start and end points
#             start = np.where(bout.global_coord >= self.params['gpos_start'])[0][0]
#             end = np.where(bout.global_coord >= self.params['gpos_end'])[0][0]
#             duration = (start - end)

#             # initialize empty arrays
#             X_batch = torch.zeros((duration, 2))  # distance, angle in polar coordinates
#             # Y_batch = torch.zeros((duration, 2))  # speed, avel

#             # # loop and get inputs/outputs every X ms
#             # for step, frame in enumerate(np.arange(start, end, self.params['n_msec_update']/1000*self.params['fps'])):
#             #     # get future frame
#             #     future = frame + int(self.params['n_msec_future'] * self.params['fps'])

#             #     # get position/orientation X ms in the future in polar coordinates
#             #     delta_x = bout.x[future] - bout.x[frame]
#             #     delta_y = bout.y[future] - bout.y[frame]
#             #     X_batch[step, 0], X_batch[step, 1] = cart2pol(delta_x, delta_y)

#             #     # get speed/avel right now
#             #     Y_batch[step, 0] = bout.speed[frame]
#                 Y_batch[step, 1] = bout.angular_velocity[frame]


#             # # RNN input: batch size * seq len * n_input
#             # X = X.reshape(1, seq_len, 1)

#             # # out shape = (batch, seq_len, num_directions * hidden_size)
#             # Y = Y.reshape(1, seq_len, 1)

#             # X_batch[:, m] = X.squeeze()
#             # Y_batch[:, m] = Y.squeeze()
#             self.items[i] = (X_batch, Y_batch)


# def make_batch(seq_len):
#     """
#     Return a single batch of given length
#     """
#     dataloader = torch.utils.data.DataLoader(
#         ThreeBitDataset(seq_len, dataset_length=1),
#         batch_size=1,
#         num_workers=0 if is_win else 2,
#         shuffle=True,
#         worker_init_fn=lambda x: np.random.seed(),
#     )

#     batch = [b for b in dataloader][0]
#     return batch


# def plot_predictions(model, seq_len, batch_size):
#     """
#     Run the model on a single batch and plot
#     the model's prediction's against the
#     input data and labels.
#     """
#     X, Y = make_batch(seq_len)
#     o, h = model.predict(X)

#     f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
#     for n, ax in enumerate(axarr):
#         ax.plot(X[0, :, n], lw=2, color=salmon, label="input")
#         ax.plot(
#             Y[0, :, n],
#             lw=3,
#             color=indigo_light,
#             ls="--",
#             label="correct output",
#         )
#         ax.plot(o[0, :, n], lw=2, color=light_green_dark, label="model output")
#         ax.set(title=f"Input {n}")
#         ax.legend()

#     f.tight_layout()
#     clean_axes(f)
