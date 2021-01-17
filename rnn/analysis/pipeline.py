from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fcutils.plotting.utils import save_figure
from myterial import orange

from pyrnn.analysis.dimensionality import get_n_components_with_pca
from pyrnn import is_win
from pyrnn._utils import npify

import sys

sys.path.append("./")

from rnn.analysis.utils import (
    load_from_folder,
    fit_fps,
    unpad,
)

from rnn.analysis._visuals import (
    plot_inputs,
    plot_outputs,
    plot_rnn_weights,
)

"""
    A standardized set of analyses to run on any RNN
"""


class Pipeline:
    X = None
    SELECT_TRIALS = False

    def __init__(
        self,
        folder,
        n_trials_in_h=128,
        interactive=False,
        fit_fps=False,
        fps_kwargs={},
        winstor=False,
    ):
        """ 
            Arguments:
                folder: Path. Path to folder with RNN data
                n_trials_in_h: int. Number of trials to use to compute h
                interactive: bool. If true the pipeline is run in interactive mode and will show renderings and images
                    interrupting the analysis for user to look at data
                fit_fps: bool. If true the fixed points of the dynamics are found
                fps_kwargs: dict. Dictionary of optional arguments for fps search
                winstor: bool. True if the pipeline is being run winstor
        """
        # set up paths and stuff
        self.folder = Path(folder)
        self.winstor = winstor

        logger.info(f"Running RNN analysis on [b {orange}]{self.folder.name}")

        self.analysis_folder = self.folder / "analysis"
        self.analysis_folder.mkdir(exist_ok=True)

        self.h_path = self.analysis_folder / "h.npy"  # hidden state
        self.X_path = self.analysis_folder / "X.npy"  # input data
        self.Y_path = self.analysis_folder / "Y.npy"  # correct output
        self.O_path = self.analysis_folder / "O.npy"  # network output

        logger.add(self.analysis_folder / "analysis_log.log")

        self.n_trials_in_h = n_trials_in_h
        self.fps_kwargs = fps_kwargs
        self.interactive = interactive
        self.fit_fps = fit_fps

    @property
    def n_trials(self):
        """
            Returns the number of trials in the data loaded
        """
        if self.X is None:
            raise ValueError(
                "Need to load data before accessing propery: n_trials"
            )
        return self.X.shape[0]

    @property
    def n_frames(self):
        """
            Returns the number of frames in the data loaded
        """
        if self.X is None:
            raise ValueError(
                "Need to load data before accessing propery: n_trials"
            )
        return self.X.shape[1]

    def setup(self, select=False):
        """
            Load necessary data for analysis and visualization

            Arguments:
                select: bool. If true trials are selected based on X[:, 0, 0]
        """
        # load RNN data
        self.dataset, self.rnn, self.fixed_points = load_from_folder(
            self.folder, winstor=self.winstor
        )

        # store some variables for easier access
        self.input_names = self.dataset.inputs_names
        self.output_names = self.dataset.outputs_names

        self.W_in = npify(self.rnn.w_in.weight)
        self.W_rec = npify(self.rnn.w_rec.weight)
        self.W_out = npify(self.rnn.w_out.weight)

        # Get/load hidden states trajectory
        self.X, self.h, self.O, self.Y = self.get_XhO()

        # not all trials are to be visualized for clarity, select some
        if select:
            self.idx_to_visualize = [
                tn for tn in range(self.X.shape[0]) if self.X[tn, 0, 0] < 0
            ]
        else:
            self.idx_to_visualize = np.arange(self.X.shape[0])

    def run(self):
        logger.debug("Running RNN analysis pipeline")
        self.setup(self.SELECT_TRIALS)

        # plot RNN I/O signal and weights
        self.plot()

        # Dymensionality analysis
        self.dimensionality()

        # fit fixed points
        if self.fit_fps:
            self.fps = fit_fps(
                self.rnn, self.h, self.analysis_folder, **self.fps_kwargs
            )

        # show plots
        if self.interactive:
            plt.show()

    def _show_save_plot(self, figure, name, _show=True):
        """
            Saves a figure to file in the analysis folder
            and if in interactive mode it shows the plot

            Arguments:
                figure: plt.Figure object
                name: str. Figure name (e.g. plot.png)
                _show: bool. If true the figure is shown to user
        """
        save_figure(
            figure, self.analysis_folder / name, verbose=False,
        )
        logger.debug(f"Saved {(self.analysis_folder / name).stem} figure")

        if self.interactive and _show:
            plt.show()
        del figure

    def get_XhO(self):
        """
            Tries to load a previously saved hidden state trajectory. 
            If there isn't one or it is of the wrong size it computes a new 
            one with self.get_h

        """
        if not self.h_path.exists():
            h, X, O, Y = self.calc_h()
        else:
            # load previously saved
            h = np.load(self.h_path)

            # check if right shape else calc anew
            if h.shape[0] != self.n_trials_in_h:
                h, X, O, Y = self.calc_h()
            else:
                logger.debug(f"Loaded h from file, shape: {h.shape}")
                X = np.load(self.X_path)
                O = np.load(self.O_path)
                Y = np.load(self.Y_path)

        return unpad(X, h, O, Y)

    def calc_h(self):
        """
            Get a trajectory of hidden states by running the network for n trials
        """
        logger.debug(
            f"Extracting hidden state trace for {self.n_trials_in_h} trials"
        )
        X, Y = self.dataset.get_one_batch(
            self.n_trials_in_h, winstor=self.winstor
        )  # get trials
        if is_win:
            X = X.cpu().to("cuda:0")

        O, h = self.rnn.predict_with_history(X)

        if np.any(np.isnan(h)) or np.any(np.isinf(h)):
            raise ValueError(
                "Found nans or infs in h, check this as it will break furher analyses"
            )

        # save and re-load to ensure everything's fine
        np.save(self.h_path, h)
        np.save(self.Y_path, Y.cpu())
        np.save(self.X_path, X.cpu())
        np.save(self.O_path, O)

        h = np.load(self.h_path)
        X = np.load(self.X_path)
        O = np.load(self.O_path)
        Y = np.load(self.Y_path)

        return h, X, O, Y

    def plot(self):
        # plot network inputs
        f = plot_inputs(
            self.X[self.idx_to_visualize, :, :], self.dataset.inputs_names
        )
        self._show_save_plot(f, "network_inputs.png", _show=False)

        # plot networks outputs
        f = plot_outputs(
            self.O[self.idx_to_visualize, :, :], self.dataset.outputs_names
        )
        self._show_save_plot(f, "network_outputs.png", _show=False)

        # plot networks weights
        f = plot_rnn_weights(self.rnn)
        self._show_save_plot(f, "network_weights.png", _show=False)

    def dimensionality(self):
        """
            Plots, renders and analysis to get at the dimensionality
            of the RNNs  dynamics

            Returns:
                dyn_dimensionality: int. Number of dimensions of the dynamics
        """
        # Look at dimensionality of hidden dynamics with PCA
        logger.debug("Getting dimensionality with PCA")
        dyn_dimensionality, f = get_n_components_with_pca(
            self.h, is_hidden=True
        )
        logger.debug(
            f"PCA says dynamics dimensionality is: {dyn_dimensionality}"
        )

        self._show_save_plot(f, "dynamics_dimensionality.png", _show=False)
        return dyn_dimensionality


if __name__ == "__main__":
    # fld = r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\RNN\trained\210113_175110_RNN_train_inout_dataset_predict_tau_from_deltaXYT"
    fld = r"Z:\swc\branco\Federico\Locomotion\control\RNN\210115_125437_RNN_lgbtch_milestones_dataset_predict_nudot_from_deltaXYT"

    fps_kwargs = dict(max_fixed_points=3, max_iters=6000, lr_decay_epoch=1500,)

    Pipeline(
        fld,
        n_trials_in_h=128,
        interactive=False,
        fit_fps=False,
        fps_kwargs=fps_kwargs,
    ).run()
