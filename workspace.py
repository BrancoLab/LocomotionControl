# %%
from proj.utils.misc import load_results_from_folder

fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/tracking_200923_141709_1491_good_example/results"

# %%
config, trajectory, history, cost_history = load_results_from_folder(fld)

# %%
import matplotlib.pyplot as plt

plt.plot(cost_history["x"])

# %%
plt.plot(history["tau_r"])
plt.plot(history["tau_l"])

# %%
plt.plot(history["v"])

# %%
