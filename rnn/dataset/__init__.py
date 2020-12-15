from myterial import salmon, salmon_dark, light_green_dark, light_green
import matplotlib.pyplot as plt
from pyrnn._plot import clean_axes


def plot_predictions(model, dataset):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = dataset._get_random()

    if model.on_gpu:
        model.cpu()
        model.on_gpu = False

    o, h = model.predict(X)

    n_inputs = X.shape[-1]
    n_outputs = Y.shape[-1]
    labels = ["x", "y", "$\\theta$", "v", "$\\omega$"]

    f, axarr = plt.subplots(nrows=2, figsize=(12, 9), sharex=True)

    for n in range(n_inputs):
        axarr[0].plot(X[0, :, n], lw=2, label=labels[n])
    axarr[0].set(title="inputs")
    axarr[0].legend()

    cc = [salmon, light_green]
    oc = [salmon_dark, light_green_dark]
    labels = ["nudot_R", "nudot_L"]
    for n in range(n_outputs):
        axarr[1].plot(
            Y[0, :, n], lw=2, color=cc[n], label="correct " + labels[n]
        )
        axarr[1].plot(
            o[0, :, n], lw=2, ls="--", color=oc[n], label="model output"
        )
    axarr[1].legend()
    axarr[1].set(title="outputs")

    f.tight_layout()
    clean_axes(f)
    return f
