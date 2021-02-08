from vedo.colors import colorMap
from loguru import logger
from vedo import Text2D, show
import numpy as np
import matplotlib.pyplot as plt

# from einops import repeat

import sys

sys.path.append("./")

from myterial import salmon

from fcutils.maths.signals import derivative
from fcutils.plot.figure import clean_axes

from pyrnn.render import render_fixed_points
from pyrnn._utils import flatten_h
from pyrnn.analysis.dimensionality import PCA


from rnn.analysis import Pipeline
from rnn.analysis._visuals import (
    render_vectors,
    COLORS,
)


class DynamicsVis(Pipeline):
    SELECT_TRIALS = False  # show only some trials?
    COLOR_BY = "var"

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

        # vmin, vmax = -0.02, 0.02

        for trialn in range(self.n_trials):
            if trialn in self.idx_to_visualize:
                if self.COLOR_BY == "var":
                    # ? color each frame in each trial
                    vmin = np.nanmin(self.X[self.idx_to_visualize, :, var])

                    D = self.X[trialn, :, var]
                    colors.append(
                        [
                            colorMap(D[i], "bwr", vmin=vmin, vmax=-vmin,)
                            for i in range(self.n_frames)
                        ]
                    )

                elif self.COLOR_BY == "outvar":
                    # ? color each frame in each trial
                    vmin = np.nanmin(self.O[self.idx_to_visualize, :, 0])
                    D = self.O[trialn, :, 0]
                    colors.append(
                        [
                            colorMap(D[i], "bwr", vmin=vmin, vmax=-vmin,)
                            for i in range(self.n_frames)
                        ]
                    )

                elif self.COLOR_BY == "trial":
                    # ? one color for the whole trial
                    vmin = np.nanmin(self.X[self.idx_to_visualize, 0, var])
                    colors.append(
                        colorMap(
                            self.X[trialn, 0, var],
                            "viridis",
                            vmin=vmin,
                            vmax=-vmin,
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
        W_in = pca.transform(self.W_in.T)
        vecs_in = render_vectors(
            [W_in[n, :] for n in range(W_in.shape[0])],
            self.input_names,
            [COLORS[l] for l in self.input_names],
            scale=0.1,
        )

        # render output weights
        W_out = pca.transform(self.W_out)
        vecs_out = render_vectors(
            [W_out[n, :] for n in range(W_out.shape[0])],
            self.output_names,
            [COLORS[l] for l in self.output_names],
            scale=50,
            showplane=False,
            showline=False,
        )

        logger.debug(
            f"Angle between read out vectors: {round(W_out[0, :].dot(W_out[1, :]), 4)}"
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

        if self.COLOR_BY == "var":
            msg = "  by variable:  " + self.input_names[var]
        elif self.COLOR_BY == "outvar":
            msg = "  by variable:  " + self.output_names[var]
        else:
            msg = ""

        logger.debug(
            f"[amber]Rendering" f"\nColoring by: {self.COLOR_BY}" + msg,
        )

        # render each trial individually
        _, acts = render_fixed_points(
            self.h[self.idx_to_visualize, :, :],
            self.fixed_points,
            alpha=0.8,
            lw=0.05,  # 0.025,
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
                + self.input_names[var]
                + f" Colored by: {self.COLOR_BY}"
                + msg,
                pos=3,
            )
        )

        # render W_in and W_out
        vectors = self.render_input_output_vectors(pca)

        # project onto readout plane
        # acts.extend(self.project_onto_plane(acts[:-1], vectors[-1]))

        return acts + vectors

    def plot_dynamics_projected_onto_readout_vectors(self):
        """
            Plots the dynamics projected onto the readout weights W_out
        """
        f, ax = plt.subplots(figsize=(16, 9))

        for trialn in range(self.n_trials):
            # compute dot product
            out = np.apply_along_axis(self.W_out.dot, 1, self.h[trialn, :, :])

            # plot
            ax.plot(out[:, 0], out[:, 1], lw=3, color=salmon, alpha=0.5)
            ax.set(
                xlabel=self.output_names[0], ylabel=self.output_names[1],
            )

        clean_axes(f)
        self._show_save_plot(f, "readout_projection.png", _show=False)

    def _plot_pca(self, pca):
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

        # plot readout directions
        out = pca.transform(self.W_out)
        scale = 200
        for n in range(out.shape[0]):
            ax.arrow(
                0,
                0,
                out[n, 0 * scale],
                out[n, 1] * scale,
                color=COLORS[self.output_names[n]],
                width=0.25,
                label=self.output_names[n],
            )

        ax.legend()
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

        # fit PCA on all data even if not rendered
        pca = PCA(n_components=3).fit(flatten_h(self.h))

        # create actors for each variable using pyrnn
        actors = []
        for var in range(self.X.shape[-1]):
            actors.append(self._render_by_var(var, pca))
            break

        # render everything in a single window
        logger.debug("Render ready")
        show(
            actors,
            N=len(actors),
            size="full",
            title="Dynamics",
            axes=4,
            interactive=self.interactive,
        ).close()

        # plot activity projected onto readout plane
        self.plot_dynamics_projected_onto_readout_vectors()

        # plot PCs
        self._plot_pca(pca)

        plt.show()


if __name__ == "__main__":
    fld = r"Z:\swc\branco\Federico\Locomotion\control\RNN\210205_174615_RNN_smallLR_dataset_predict_PNN_from_RPsy"

    # TODO figure out why angle of W_out projections looks different in 2 and 3 dimensions

    DynamicsVis(
        fld, n_trials_in_h=128, fit_fps=False, interactive=True
    ).visualize()
