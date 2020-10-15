import matplotlib.pyplot as plt
import numpy as np
import logging

from fcutils.plotting.utils import clean_axes
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_line_outlined

from proj.animation import variables_colors as colors


def press(event, self):
    """ 
        Deals with key press during interactive visualizatoin
    """
    if event.key == "c":
        logging.info(
            "Stopping because user manually terminated simulation (presed C)"
        )
        self.stop = True


class Plotter:
    def make_figure(self):
        plt.ion()

        self.f = plt.figure(figsize=(16, 8))

        gs = self.f.add_gridspec(2, 3)  # 6)
        self.xy_ax = self.f.add_subplot(gs[:, :2])
        self.xy_ax.axis("equal")
        self.xy_ax.axis("off")

        self.tau_ax = self.f.add_subplot(gs[0, 2:4])

        self.sax = self.f.add_subplot(gs[1, 2:4])

        # self.accel_ax = self.f.add_subplot(gs[0, 4])

        # self.goal_ax = self.f.add_subplot(gs[0, 5])

        # self.cost_ax = self.f.add_subplot(gs[1, 4:])

        clean_axes(self.f)

        self.f.canvas.mpl_connect(
            "key_press_event", lambda event: press(event, self)
        )

    # ------------------------------- Live plotting ------------------------------ #
    def _plot_xy(self, ax, curr_goals):
        # plot currently selected goals
        if self.model.MODEL_TYPE == "cartesian":
            ax.plot(
                curr_goals[:, 0],
                curr_goals[:, 1],
                lw=10,
                color="r",
                alpha=0.5,
                zorder=-1,
                solid_capstyle="round",
            )
        else:
            print("current goals for polar")
            # raise NotImplementedError("current goals for polar")

        # plot position history
        X = [pos.x for pos in self.model_position_history_world]
        Y = [pos.y for pos in self.model_position_history_world]

        ax.plot(
            X,
            Y,
            lw=9,
            color=desaturate_color(colors["tracking"]),
            zorder=-1,
            solid_capstyle="round",
        )

        # plot current position
        x, y = self.model_position_world.x, self.model_position_world.y

        ax.scatter(  # plot body
            x,
            y,
            s=200,
            color=colors["tracking"],
            lw=1.5,
            edgecolors=[0.3, 0.3, 0.3],
        )

        if self.model.MODEL_TYPE == "cartesian":
            # plot body axis
            t = self.model_position_world.t
            dx = np.cos(t) * (
                self.model.mouse["length"]
                * (1 / self.model.trajectory["px_to_cm"])
                - 0.5
            )
            dy = np.sin(t) * (
                self.model.mouse["length"]
                * (1 / self.model.trajectory["px_to_cm"])
            )

            ax.plot([x, x + dx], [y, y + dy], lw=8, color=colors["tracking"])
            ax.scatter(  # plot head
                x + dx,
                y + dy,
                s=125,
                color=colors["tracking"],
                lw=1.5,
                edgecolors=[0.3, 0.3, 0.3],
            )

        ax.axis("equal")
        ax.axis("off")

    def _plot_control(self, keep_s=1.2):
        keep_n = int(keep_s / self.model.dt)
        ax = self.tau_ax
        ax.clear()

        R, L = self.model.history["tau_r"], self.model.history["tau_l"]
        n = len(R)

        # plot traces
        plot_line_outlined(
            ax,
            R,
            color=colors["tau_r"],
            label="$\\tau_R$",
            lw=2,
            solid_joinstyle="round",
            solid_capstyle="round",
        )
        plot_line_outlined(
            ax,
            L,
            color=colors["tau_l"],
            label="$\\tau_L$",
            lw=2,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

        # set axes
        ymin = np.min(np.vstack([R[n - keep_n : n], L[n - keep_n : n]]))
        ymax = np.max(np.vstack([R[n - keep_n : n], L[n - keep_n : n]]))

        if n > keep_n:
            ymin -= np.abs(ymin) * 0.1
            ymax += np.abs(ymax) * 0.1

            ax.set(xlim=[n - keep_n, n], ylim=[ymin, ymax])

        ax.set(ylabel="Torque\n($\\frac{cm^2 g}{s^2}$)", xlabel="step n")
        ax.legend()
        ax.set(title="Control")

        # mark the times a change of traj idx happened
        # for v in self.moved_to_next:
        #     ax.axvline(v, color="k")

    def _plot_current_variables(self):
        """
            Plot the agent's current state vs where it should be
        """

        ax = self.sax
        ax.clear()

        # plot speed trajectory

        if self.model.MODEL_TYPE == "cartesian":
            idx = 3
        else:
            idx = 2

        ax.scatter(
            np.arange(len(self.initial_trajectory[:, idx]))[
                :: self.plot_every
            ],
            self.initial_trajectory[:, idx][:: self.plot_every],
            color=colors["v"],
            label="trajectory speed",
            lw=1,
            edgecolors=[0.8, 0.8, 0.8],
            s=100,
        )

        # plot current speed
        ax.scatter(
            self.curr_traj_waypoint_idx,
            self.model.history["v"][-1],
            zorder=100,
            s=300,
            lw=1,
            color=colors["v"],
            edgecolors="k",
            label="models speed",
        )

        # store the scatter coords for later plots
        self._cache["speed_plot_x"].append(self.curr_traj_waypoint_idx)
        self._cache["speed_plot_y"].append(self.model.history["v"][-1])

        # plot line
        ax.plot(
            self._cache["speed_plot_x"],
            self._cache["speed_plot_y"],
            color=desaturate_color(colors["v"]),
            zorder=-1,
            lw=9,
        )

        ax.legend()
        ax.set(title="Speed")
        ax.set(ylabel="speed", xlabel="trajectory progression")

    def _plot_accelerations(self):
        ax = self.accel_ax
        ax.clear()

        ax.bar(
            [0, 1, 2, 3],
            [
                self.model.curr_dxdt.v_dot,
                self.model.curr_dxdt.omega_dot,
                self.model.curr_wheel_state.nudot_left,
                self.model.curr_wheel_state.nudot_right,
            ],
            color=[
                colors["v"],
                colors["omega"],
                desaturate_color(colors["tau_l"]),
                desaturate_color(colors["tau_r"]),
            ],
        )
        ax.set(
            xticklabels=[
                "$\dot{v}$",
                "$\dot{\omega}$",
                "$\dot{\\nu_R}$",
                "$\dot{\\nu_L}$",
            ],
            xticks=[0, 1, 2, 3],
        )
        ax.set(title="Acceleration")

    def _plot_cost(self, keep_s=1.2):
        keep_n = int(keep_s / self.model.dt)

        ax = self.cost_ax
        ax.clear()

        for k, v in self.cost_history.items():
            if "total" not in k:
                n = len(v)
                if n > keep_n + 2:
                    x = v[n - keep_n :]
                else:
                    x = v.copy()

                ax.plot(
                    x, label=k, lw=3, solid_capstyle="round", color=colors[k],
                )
        ax.legend()
        ax.set(title="Cost")

    def _plot_goal(self, goal):
        self.goal_ax.clear()
        goal = self.model._state(*goal)
        x = self.model.curr_x

        for n, k in enumerate(x._fields):
            self.goal_ax.bar(
                n,
                goal._asdict()[k] - x._asdict()[k],
                color=desaturate_color(colors[k]),
                zorder=99,
                lw=1,
                ec="k",
                label=k,
            )

        self.goal_ax.legend()
        self.goal_ax.set(xticks=[])
        self.goal_ax.set(title="Goal")

    def visualize_world_live(self, curr_goals, elapsed=None):
        ax = self.xy_ax
        ax.clear()

        # plot trajectory
        ax.scatter(
            self.initial_trajectory[:: self.plot_every, 0],
            self.initial_trajectory[:: self.plot_every, 1],
            s=50,
            color=colors["trajectory"],
            lw=1,
            edgecolors=[0.8, 0.8, 0.8],
        )

        # highlight current trajectory point
        ax.scatter(
            self.current_traj_waypoint[0],
            self.current_traj_waypoint[1],
            s=30,
            color="r",
            lw=1,
            edgecolors=[0.8, 0.8, 0.8],
            zorder=99,
        )

        # plot XY tracking
        self._plot_xy(ax, curr_goals)
        ax.set(
            title=f"Elapsed time: {round(elapsed, 2)}s | goal: {round(self.goal_duration, 2)}s\n"
            + f"trajectory progression: {self.curr_traj_waypoint_idx}/{len(self.initial_trajectory)}\n"
            + f'Curr cost: {round(self.curr_cost["total"], 2)} | total: {round(self.total_cost, 2)}'
        )

        # plot control
        self._plot_control()

        # plot sped
        self._plot_current_variables()

        # # plot accelerations
        # self._plot_accelerations()

        # # plot cost
        # self._plot_cost()

        # # plot goal
        # self._plot_goal(curr_goals[0, :])

        # display plot
        self.f.canvas.draw()
        plt.pause(0.00001)

        # save figure for gif making
        if self.itern < 10:
            n = f"0{self.itern}"
        else:
            n = str(self.itern)
        self.f.savefig(str(self.frames_folder / n))
