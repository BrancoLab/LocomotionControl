import matplotlib.pyplot as plt

def plot_trajectory(traj):

    f, ax = plt.subplots(figsize=(16,9))

    ax.scatter(traj[:, 0], traj[:, 1], color=traj[:, 2], lw=1, edgecolors='k')