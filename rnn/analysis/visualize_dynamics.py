import matplotlib.pyplot as plt
import numpy as np
import sys
from loguru import logger

from fcutils.maths.coordinates import pol2cart


sys.path.append("./")
from rnn.analysis import Pipeline
from rnn.analysis._visuals import COLORS


class DynamicsVis(Pipeline):
    def __init__(
        self,
        folder,
        n_trials_in_h=128,
        interactive=False,
        fit_fps=False,
        fps_kwargs={},
    ):
        """ 
                Arguments:
                    folder: Path. Path to folder with RNN data
                    n_trials_in_h: int. Number of trials to use to compute h
                    interactive: bool. If true the pipeline is run in interactive mode and will show renderings and images
                        interrupting the analysis for user to look at data
                    fit_fps: bool. If true the fixed points of the dynamics are found
                    fps_kwargs: dict. Dictionary of optional arguments for fps search
            """
        Pipeline.__init__(
            self,
            folder,
            n_trials_in_h=n_trials_in_h,
            interactive=interactive,
            fit_fps=fit_fps,
            fps_kwargs=fps_kwargs,
        )

        # set things up
        self.setup()
        self.make_figure()

        # unscale data
        self.X_, self.Y_ = self.unscale_data()

        # fit PCA on all data
        # self.pca = PCA(n_components=3).fit(flatten_h(self.h))

        # plot things
        self.plot_xy()
        self.plot_inputs()
        self.plot_outputs()

    def unscale_data(self):
        """
            Undoes the scaling of input/output data used for
            training the RNNs
        """
        logger.debug("Unscaling data")
        # load normalizers
        train_normalizer, test_normalizer = self.dataset.load_normalizers()

        # stack data
        data = np.zeros(
            (self.X.shape[0], self.X.shape[1], self.ninputs + self.noutputs)
        )
        data[:, :, : self.ninputs] = self.X.copy()
        data[:, :, self.ninputs :] = self.Y.copy()

        # scale
        scaled = np.zeros_like(data)
        for trial in range(data.shape[0]):
            scaled[trial, :, :] = self.dataset.unscale(
                data[trial], train_normalizer
            )

        # unpad
        X_ = scaled[:, :, : self.ninputs]
        Y_ = scaled[:, :, self.ninputs :]

        X_[self.X == np.nan] = np.nan
        Y_[self.Y == np.nan] = np.nan
        return X_, Y_

    def make_figure(self):
        """
            Create a figure to visualize the data in
        """

        f = plt.figure(figsize=(10, 10))
        gs = f.add_gridspec(
            ncols=4, nrows=2 + max(self.ninputs, self.noutputs)
        )
        self.xy_ax = f.add_subplot(gs[:2, :2], aspect="equal")
        self.pca_ax = f.add_subplot(gs[:2, 2:], projection="3d")

        self.input_axes = [
            f.add_subplot(gs[2 + n, :2]) for n in range(self.ninputs)
        ]
        self.output_axes = [
            f.add_subplot(gs[2 + n, 2:]) for n in range(self.noutputs)
        ]

        self.xy_ax.set(xlabel="cm", ylabel="cm")

        f.tight_layout()

    def plot_xy(self):
        """
            Plot the XY tracking for the input data, if the dataset uses
            polar inputs , these have to be converted to cartesian coordinates
        """
        # check arguments names
        if self.dataset.polar:
            if self.input_names[0] != "r" or self.input_names[1] != "psy":
                NotImplementedError("Unrecognized inputs names")
        else:
            if self.input_names[0] != "x" or self.input_names[1] != "y":
                NotImplementedError("Unrecognized inputs names")

        # plot each trial
        for trial in range(self.n_trials_in_h):
            if self.dataset.polar:
                x, y = pol2cart(self.X_[trial, :, 0], self.X_[trial, :, 1])
            else:
                x, y = self.X_[trial, :, 0], self.X_[trial, :, 1]

            self.xy_ax.plot(x, y, lw=1, color=[0.2, 0.2, 0.2])

    def plot_inputs(self):
        for n, name in enumerate(self.input_names):
            self.input_axes[n].plot(
                self.X_[:, :, n].T, color=COLORS[name], lw=1, alpha=0.6
            )
            self.input_axes[n].set(xlabel="frame", ylabel=name)

    def plot_outputs(self):
        for n, name in enumerate(self.output_names):
            self.output_axes[n].plot(
                self.Y_[:, :, n].T, color=COLORS[name], lw=1, alpha=0.6
            )
            self.output_axes[n].set(xlabel="frame", ylabel=name)


if __name__ == "__main__":
    fld = r"Z:\swc\branco\Federico\Locomotion\control\RNN\210209_143926_RNN_short_dataset_predict_PNN_from_deltaXYT"
    DynamicsVis(fld, n_trials_in_h=1, fit_fps=False, interactive=True)
    plt.show()
