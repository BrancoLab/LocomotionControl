from loguru import logger
from pathlib import Path
import numpy as np

from fcutils.plotting.utils import save_figure

from pyrnn.analysis.dimensionality import get_n_components_with_pca

# from pyrnn.render import render_state_history_pca_3d
from pyrnn import is_win

import sys

sys.path.append("./")

from rnn.analysis.utils import load_from_folder

"""
    A standardized set of analyses to run on any RNN
"""


class Pipeline:
    def __init__(self, folder, n_trials_in_h=128):
        """ 
            Arguments:
                folder: Path. Path to folder with RNN data
                n_trials_in_h: int. Number of trials to use to compute h
        """
        self.folder = Path(folder)
        self.analysis_folder = self.folder / "analysis"
        self.analysis_folder.mkdir(exist_ok=True)

        logger.add(self.analysis_folder / "log.log")

        self.n_trials_in_h = n_trials_in_h

        # TODO if in interactive mode make 3d renderings with vedo too

    def run(self):
        logger.debug("Running RNN analysis pipeline")

        # load RNN data
        self.dataset, self.rnn = load_from_folder(self.folder)

        # Get hidden states trajectory
        self.h = self.get_h()

        # Dymensionality analysis
        self.dimensionality()

    def get_h(self):
        """
            Get a trajectory of hidden states
        """
        logger.debug("Extracting hidden state trace")
        X, _ = self.dataset.get_one_batch(self.n_trials_in_h)  # get trials
        if is_win:
            X = X.cpu().to("cuda:0")
        _, h = self.rnn.predict_with_history(X)
        np.save(self.analysis_folder / "h.npy", h)
        return h

    def dimensionality(self):
        """
            Plots and analysis to get at the dimensionality
            of both an RNNs connectivity matrix and of its dynamics
        """
        # Look at dimensionality of hidden dynamics
        dyn_dimensionality, f = get_n_components_with_pca(
            self.h, is_hidden=True
        )
        logger.debug(
            f"PCA says dynamics dimensionality is: {dyn_dimensionality}"
        )
        save_figure(
            f,
            self.analysis_folder / "dynamics_dimensionality.png",
            verbose=False,
        )
        logger.debug("Saved dynamics dim. figure")
        del f


if __name__ == "__main__":
    fld = r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\control\RNN\trained\201221_170210_RNN_delta_dataset_predict_tau_from_deltaXYT"
    Pipeline(fld, n_trials_in_h=1).run()
