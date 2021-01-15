from vedo.colors import colorMap
from loguru import logger
from vedo import Text2D, show
import numpy as np
import matplotlib.pyplot as plt

# from einops import repeat

import sys

sys.path.append("./")

from fcutils.maths.utils import derivative
from fcutils.plotting.utils import clean_axes

from pyrnn.render import render_state_history_pca_3d
from pyrnn._utils import npify, flatten_h
from pyrnn.analysis.dimensionality import PCA


from rnn.analysis import Pipeline
from rnn.analysis._visuals import (
    render_vectors,
    COLORS,
)


class DynamicsVis(Pipeline):
    SELECT_TRIALS = False  # show only some trials?
    COLOR_BY = "speed"
    START = 0  # first frame to show
    STOP = -1  # last frame to show

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

    def _get_trial_colors(self, var):
        """
            Gets a list of colors to color the actors of each trial. 
            Each trial can be assigned a single color or a list of colors
            with one color for each frame in the trial.

            Arguments:
                var: int. Index of variable to be used for setting colors
        """
        colors = []
        vmin = np.nanmin(self.X[self.idx_to_visualize, 0, var])
        vmax = np.nanmax(self.X[self.idx_to_visualize, 0, var])

        # vmin, vmax = -0.02, 0.02

        for trialn in range(self.n_trials):
            if trialn in self.idx_to_visualize:
                if self.COLOR_BY == "var":
                    # ? color each frame in each trial
                    D = derivative(self.X[trialn, :, 2])
                    colors.append(
                        [
                            colorMap(D[i], "bwr", vmin=vmin, vmax=vmax,)
                            for i in range(self.n_frames)
                        ]
                    )

                elif self.COLOR_BY == "trial":
                    # ? one color for the whole trial
                    colors.append(
                        colorMap(
                            self.X[trialn, 0, var],
                            "viridis",
                            vmin=vmin,
                            vmax=vmax,
                        )
                    )

                elif self.COLOR_BY == "time":
                    # ? color by time
                    colors.append(
                        [
                            colorMap(i, "bwr", vmin=0, vmax=self.n_frames,)
                            for i in range(self.n_frames)
                        ]
                    )

                elif self.COLOR_BY == "speed":
                    # ? color by speed of the dynamics
                    speed = np.abs(
                        np.nanmean(
                            derivative(
                                np.nan_to_num(self.h[trialn, :, :]), axis=0
                            ),
                            axis=1,
                        )
                    )
                    colors.append(
                        [colorMap(i, "bwr", vmin=0, vmax=0.015) for i in speed]
                    )

        return colors

    def project_onto_plane(self, actors, plane):
        """
            Creates projections of single actors onto a given plane.

            Arguments:
                actors: list of actors
                plane: Plane actor to project onto

            Returns:
                projections: list of actors with projections
        """
        projections = []
        for act in actors:
            projections.append(
                act.clone().projectOnPlane(plane).c("k").alpha(1)
            )
        return projections

    def render_input_output_vectors(self, pca):
        """
            Renders the PCA embedding of W_in 
            and W_out weights matrices.

            Arguments:
                pca: PCA fitted to self.h

            Returns:
                actors: list of actors with vectors As Arrows
        """
        logger.debug("Rendering W_in and W_out")

        # render input weights as vectors
        W_in = pca.transform(npify(self.rnn.w_in.weight).T)
        vecs_in = render_vectors(
            [W_in[0, :], W_in[1, :], W_in[2, :]],
            self.dataset.inputs_names,
            [COLORS[l] for l in self.dataset.inputs_names],
        )

        # render output weights
        W_out = pca.transform(npify(self.rnn.w_out.weight))
        vecs_out = render_vectors(
            [W_out[0, :], W_out[1, :]],
            self.dataset.outputs_names,
            [COLORS[l] for l in self.dataset.outputs_names],
            showplane=True,
            showline=False,
        )

        logger.debug(
            f"Angle between read out vectors: {round(np.degrees(W_out[0, :].dot(W_out[1, :])), 4)}"
        )

        return vecs_in + vecs_out

    def _render_by_var(self, var, pca):
        """
            Create all actors to visualize dynamics colored according
            to the values of an input variable

            Arguments:
                var: int. Index of variable to use
                pca: PCA model fit to self.h
            
            Returns
                actors: list of actors.
        """
        colors = self._get_trial_colors(var)

        logger.debug(
            f"[amber]Rendering colored by variable {self.dataset.inputs_names[var]}."
            f"\nRendering frames in range {self.START}-{self.STOP}",
            f"\nColoring by: {self.COLOR_BY}",
        )

        # render each trial individually
        _, acts = render_state_history_pca_3d(
            self.h[self.idx_to_visualize, self.START : self.STOP, :],
            alpha=0.8,
            lw=0.025,
            mark_start=True,
            start_color="r",
            _show=False,
            color=colors,
            color_by_trial=True,
            pca=pca,
        )

        # render variable name
        acts.append(
            Text2D(
                "var: "
                + self.dataset.inputs_names[var]
                + f" Colored by: {self.COLOR_BY}",
                pos=3,
            )
        )

        # render W_in and W_out
        vectors = self.render_input_output_vectors(pca)

        return acts + vectors

    def _plot(self, pca):
        """
            Plots the network's dynamics in 2D and each PC independently

            Arguments:
                pca: PCA with n_components=2 fitted to self.h
        """
        # plot each PC in its own subplot
        f, axarr = plt.subplots(nrows=pca.n_components, figsize=(16, 9))
        f.suptitle("All PCs")

        for pc_n in range(pca.n_components):
            for trialn in range(self.n_trials):
                pcs = pca.transform(self.h[trialn, :, :])
                axarr[pc_n].plot(
                    pcs[:, pc_n], color=[0.3, 0.3, 0.3], alpha=0.7
                )
            axarr[pc_n].set(
                title=f"PC: {pc_n}", xlabel="frames", ylabel="val."
            )

        clean_axes(f)
        self._show_save_plot(f, "all_PCs.png", _show=False)

        # plot first two PCs
        if pca.n_components < 2:
            return

        f, ax = plt.subplots(figsize=(16, 9))
        f.suptitle("First two PCs")
        for trialn in range(self.n_trials):
            pcs = pca.transform(self.h[trialn, :, :])
            ax.plot(pcs[:, 0], pcs[:, 1], color=[0.3, 0.3, 0.3], alpha=0.7)

        ax.set(xlabel="PC1", ylabel="PC2")

        clean_axes(f)
        self._show_save_plot(f, "top_2_PCs.png", _show=False)

    def visualize(self):
        """
            Renders the dynamics in 3D PCA space, coloring
            the trials based on the values of each variable in X
        """
        logger.debug(f"Rendering dynamics")

        # load data
        self.setup(select=self.SELECT_TRIALS)

        # create renderings or plots based on dimensionality of dynamics
        if self.dimensionality() > 2:
            # fit PCA on all data even if not rendered
            pca, _ = render_state_history_pca_3d(self.h, _show=False)

            # create actors for each variable using pyrnn
            actors = []
            for var in range(self.X.shape[-1]):
                actors.append(self._render_by_var(var, pca))
                break

            # render everything in a single window
            logger.debug("Render ready")
            show(
                actors, N=len(actors), size="full", title="Dynamics", axes=4
            ).close()

            # plot PCs independently
            self._plot(pca)

        else:
            # fit PCA on all daata
            pca = PCA(n_components=2).fit(flatten_h(self.h))

            # create plots
            self._plot(pca)


if __name__ == "__main__":
    fld = r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\RNN\trained\210113_175110_RNN_train_inout_dataset_predict_tau_from_deltaXYT"
    DynamicsVis(
        fld, n_trials_in_h=256, fit_fps=False, interactive=True
    ).visualize()
