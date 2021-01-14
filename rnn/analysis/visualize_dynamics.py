from vedo.colors import colorMap
from loguru import logger
from vedo import Text2D, show
import numpy as np

# from einops import repeat

import sys

sys.path.append("./")

from fcutils.maths.utils import derivative

from pyrnn.render import render_state_history_pca_3d
from pyrnn._utils import npify


from rnn.analysis import Pipeline
from rnn.analysis.utils import (
    render_vectors,
    COLORS,
)


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

        # load data
        self.setup()

    def _get_trial_colors(self, var):
        """
            Gets a list of colors to color the actors of each trial. 
            Each trial can be assigned a single color or a list of colors
            with one color for each frame in the trial.

            Arguments:
                var: int. Index of variable to be used for setting colors
        """
        colors = []
        # vmin = np.nanmin(self.X[self.idx_to_visualize, 0, var])
        # vmax = np.nanmax(self.X[self.idx_to_visualize, 0, var])

        vmin, vmax = -0.2, 0.2

        for trialn in range(self.X.shape[0]):
            if trialn in self.idx_to_visualize:
                D = derivative(self.X[trialn, :, 2])

                # color each frame in each trial
                colors.append(
                    [
                        colorMap(D[i], "viridis", vmin=vmin, vmax=vmax,)
                        for i in range(self.X.shape[1])
                    ]
                )

                # # one color for the whole trial
                # colors.append(
                #     colorMap(
                #         self.X[trialn, 0, var], "viridis", vmin=vmin, vmax=vmax,
                #     )
                # )

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

        STOP = 400  # show until this frame

        # render each trial individually
        _, acts = render_state_history_pca_3d(
            self.h[self.idx_to_visualize, :STOP, :],
            alpha=0.8,
            lw=0.025,
            mark_start=True,
            start_color="r",
            _show=False,
            color=colors,
            color_by_trial=True,
            pca=pca,
        )

        # # render mean trace
        # H = np.nanmean(self.h[self.idx_to_visualize, 1:, :], 0)[:STOP, :]

        # _, H_acts = render_state_history_pca_3d(
        #     repeat(H, "n i -> b n i", b=1),
        #     alpha=1,
        #     lw=0.05,
        #     color='k',
        #     pca=pca,
        #     _show=False,
        # )

        # acts += H_acts

        # render variable name
        acts.append(Text2D(self.dataset.inputs_names[var], pos=3))

        # render W_in and W_out
        vectors = self.render_input_output_vectors(pca)

        return acts + vectors

    def visualize(self):
        """
            Renders the dynamics in 3D PCA space, coloring
            the trials based on the values of each variable in X
        """
        logger.debug(f"Rendering dynamics")

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


# TODO remove warmup time from predictions ?

if __name__ == "__main__":
    fld = r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\RNN\trained\201221_170210_RNN_delta_dataset_predict_tau_from_deltaXYT"
    dv = DynamicsVis(fld, n_trials_in_h=10, fit_fps=False, interactive=True,)
    # dv.plot()
    dv.visualize()
