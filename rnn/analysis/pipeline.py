from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fcutils.plotting.utils import save_figure

from pyrnn.analysis.dimensionality import get_n_components_with_pca
from pyrnn import is_win
from pyrnn._utils import npify

import sys

sys.path.append("./")

from rnn.analysis.utils import (
    load_from_folder,
    fit_fps,
    plot_inputs,
    plot_outputs,
    unpad,
)

"""
    A standardized set of analyses to run on any RNN
"""


class Pipeline:
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
        # set up paths and stuff
        self.folder = Path(folder)
        self.analysis_folder = self.folder / "analysis"
        self.analysis_folder.mkdir(exist_ok=True)

        self.h_path = self.analysis_folder / "h.npy"  # hidden state
        self.X_path = self.analysis_folder / "X.npy"  # input data
        self.O_path = self.analysis_folder / "O.npy"  # network output

        logger.add(self.analysis_folder / "analysis_log.log")

        self.n_trials_in_h = n_trials_in_h
        self.fps_kwargs = fps_kwargs
        self.interactive = interactive
        self.fit_fps = fit_fps

    def run(self):
        logger.debug("Running RNN analysis pipeline")

        # load RNN data
        self.dataset, self.rnn = load_from_folder(self.folder)

        # Get/load hidden states trajectory
        self.X, self.h, self.O = self.get_XhO()

        # not all trials are to be visualized for clarity, select some
        # self.idx_to_visualize = [tn for tn in range(self.X.shape[0]) if self.X[tn, 0, 0] < 0]
        self.idx_to_visualize = np.arange(self.X.shape[0])

        # make some plots
        self.make_plots()

        # Dymensionality analysis
        self.dimensionality()

        # Visualize dynamics
        if self.interactive:
            self.render_dynamics()

        # fit fixed points
        if self.fit_fps:
            self.fps = fit_fps(
                self.rnn, self.h, self.analysis_folder, **self.fps_kwargs
            )

    def _show_save_plot(self, figure, path):
        """
            Saves a figure to file in the analysis folder
            and if in interactive mode it shows the plot
        """
        save_figure(
            figure, self.analysis_folder / path, verbose=False,
        )
        logger.debug(f"Saved {(self.analysis_folder / path).stem} figure")

        if self.interactive:
            plt.show()
        del figure

    def get_XhO(self):
        """
            Tries to load a previously saved hidden state trajectory. 
            If there isn't one or it is of the wrong size it computes a new 
            one with self.get_h

        """
        if not self.h_path.exists():
            return self.get_h()

        h = np.load(self.h_path)
        if h.shape[0] != self.n_trials_in_h:
            h, X, O = self.calc_h()
        else:
            logger.debug(f"Loaded h from file, shape: {h.shape}")
            X = np.load(self.X_path)
            O = np.load(self.O_path)
        return unpad(X, h, O)

    def calc_h(self):
        """
            Get a trajectory of hidden states by running the network for n trials
        """
        logger.debug(
            f"Extracting hidden state trace for {self.n_trials_in_h} trials"
        )
        X, _ = self.dataset.get_one_batch(self.n_trials_in_h)  # get trials
        if is_win:
            X = X.cpu().to("cuda:0")

        O, h = self.rnn.predict_with_history(X)

        if np.any(np.isnan(h)) or np.any(np.isinf(h)):
            raise ValueError(
                "Found nans or infs in h, check this as it will break furher analyses"
            )

        np.save(self.h_path, h)
        np.save(self.X_path, X.cpu())
        np.save(self.O_path, O)

        return h, npify(X), O

    def make_plots(self):
        # plot network inputs
        f = plot_inputs(
            self.X[self.idx_to_visualize, :, :], self.dataset.inputs_names
        )
        self._show_save_plot(f, "network_inputs.png")

        # plot networks outputs
        f = plot_outputs(
            self.O[self.idx_to_visualize, :, :], self.dataset.outputs_names
        )
        self._show_save_plot(f, "network_outputs.png")

        pass

    def dimensionality(self):
        """
            Plots, renders and analysis to get at the dimensionality
            of the RNNs  dynamics
        """
        # Look at dimensionality of hidden dynamics with PCA
        logger.debug("Getting dimensionality with PCA")
        dyn_dimensionality, f = get_n_components_with_pca(
            self.h, is_hidden=True
        )
        logger.debug(
            f"PCA says dynamics dimensionality is: {dyn_dimensionality}"
        )

        self._show_save_plot(f, "dynamics_dimensionality.png")
        if self.interactive:
            plt.show()


if __name__ == "__main__":
    fld = r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\RNN\trained\201221_170210_RNN_delta_dataset_predict_tau_from_deltaXYT"
    Pipeline(fld, n_trials_in_h=256, interactive=True, fit_fps=False).run()
