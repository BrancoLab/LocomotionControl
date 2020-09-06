import matplotlib.pyplot as plt

from fcutils.plotting.utils import clean_axes

from proj.utils import load_results_from_folder

def plot_results(results_folder):
    config, control, state, trajectory, history = load_results_from_folder(results_folder)

    f, axarr = plt.subplots(nrows=3, figsize=(16, 9))

    axarr[0].plot(trajectory[:, 0], trajectory[:, 1], lw=2, color=[.6, .6, .6])
    
    axarr[0].plot(history['x'], history['y'], color='g', lw=3, ls='--')

    axarr[1].plot(history['tau_r'], color='b', lw=3, ls='--')
    axarr[1].plot(history['tau_l'], color='r', lw=3, ls='--')

    axarr[0].axis('equal')


    clean_axes(f)