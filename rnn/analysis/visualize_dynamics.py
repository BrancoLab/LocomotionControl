import matplotlib.pyplot as plt
import numpy as np
import sys
from loguru import logger
from vedo import show, Spheres

from pyrnn._utils import flatten_h
from pyrnn.analysis.dimensionality import PCA


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
        self.pca = PCA(n_components=3).fit(flatten_h(self.h))

        # plot tracking inputs and outputs
        self.plot_xy()
        self.plot_inputs()
        self.plot_outputs()

        # plot neural data
        self.plot_trials_pc()

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
            (
                self.X.shape[0],
                self.X.shape[1],
                2 + self.ninputs + self.noutputs,
            )
        )
        data[:, :, : self.ninputs] = self.X.copy()
        data[:, :, self.ninputs : -2] = self.Y.copy()
        data[:, :, -2:] = self.tracking[:, self.START : self.STOP, :]

        # scale
        scaled = np.zeros_like(data)
        for trial in range(data.shape[0]):
            scaled[trial, :, :] = self.dataset.unscale(
                data[trial], train_normalizer
            )

        # unpad
        X_ = scaled[:, :, : self.ninputs]
        Y_ = scaled[:, :, self.ninputs : -2]

        X_[self.X == np.nan] = np.nan
        Y_[self.Y == np.nan] = np.nan
        self.tracking = scaled[:, :, -2:]
        return X_, Y_

    def make_figure(self):
        """
            Create a figure to visualize the data in
        """

        f = plt.figure(figsize=(14, 14))
        gs = f.add_gridspec(
            ncols=6, nrows=4 + max(self.ninputs, self.noutputs)
        )
        self.xy_ax = f.add_subplot(gs[:4, :3], aspect="equal")
        self.pca_ax = f.add_subplot(gs[:4, 3:], projection="3d")

        self.input_axes = [
            f.add_subplot(gs[4 + n, :3]) for n in range(self.ninputs)
        ]
        self.output_axes = [
            f.add_subplot(gs[4 + n, 3:]) for n in range(self.noutputs)
        ]

        self.xy_ax.set(xlabel="cm", ylabel="cm")
        self.pca_ax.set(
            xlabel="PC1",
            ylabel="PC2",
            zlabel="PC3",
            xlim=[-3, 3],
            ylim=[-3, 3],
            zlim=[-3, 3],
        )

        f.tight_layout()

    def plot_xy(self):
        """
            Plot the XY tracking for the input data, if the dataset uses
            polar inputs , these have to be converted to cartesian coordinates
        """
        # plot each trial
        for trial in range(self.n_trials_in_h):
            self.xy_ax.plot(
                self.tracking[trial, :, 0],
                self.tracking[trial, :, 1],
                lw=1,
                color=[0.2, 0.2, 0.2],
            )

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

    def plot_trials_pc(self, every=10):
        PC = self.pca.transform(self.h)  # n triials x n frames x n components

        actors = []
        c = np.arange(len(PC[0, ::every, 2]))
        for trial in range(PC.shape[0]):
            self.pca_ax.scatter(
                PC[trial, ::every, 0],
                PC[trial, ::every, 1],
                PC[trial, ::every, 2],
                c=c,
            )
            actors.append(Spheres(PC[trial, ::every, :], r=0.1, c=trial))

        plt.show()
        plotter = show(actors, size="full", interactive=True, axes=3)
        plotter.close()


if __name__ == "__main__":
    fld = r"Z:\swc\branco\Federico\Locomotion\control\RNN\210209_143926_RNN_short_dataset_predict_PNN_from_deltaXYT"
    DynamicsVis(fld, n_trials_in_h=60, fit_fps=False, interactive=True)
    plt.show()
