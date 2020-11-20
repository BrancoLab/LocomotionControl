# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from proj.rnn._utils import RNNPaths

# %%

ut = RNNPaths(dataset_name="dataset_predict_nudot")

data = pd.read_hdf(ut.dataset_train_path)

data.head()
# %%
vrs = dict(x=[], y=[], theta=[], v=[], omega=[])
names = list(vrs.keys())
for i, t in data.iterrows():
    for n, name in enumerate(names):
        vrs[name].extend(list(t.trajectory[:, n]))

# %%
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(16, 9))
axarr = axarr.flatten()
b = np.linspace(-1, 1, 20)

for n, name in enumerate(names):
    axarr[n].hist(vrs[name], bins=b, density=True)
    axarr[n].set(title=name)
# %%
