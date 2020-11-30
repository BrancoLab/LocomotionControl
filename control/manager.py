from loguru import logger
import numpy as np
import json
from rich import print
import shutil
from pyrnn._progress import base_progress as progress
from pyrnn._utils import GracefulInterruptHandler

from ._io import send_slack_message, DropBoxUtils, upload_folder
from .live_plot import Plotter
from .plot import plot_results, animate_from_images
from .history import History
from .config import dt, MANAGER_CONFIG, TRAJECTORY_CONFIG, all_configs
from control import paths

from .world import World
from .control import Controller
from .model import Model


class Manager:
    def __init__(self, winstor=False):
        self.winstor = winstor

        # Set up
        self.setup_paths()
        self.start_logging()

        # Set up classes
        self.history = History()
        self.world = World(self.trials_cache)
        self.model = Model()
        self.controller = Controller(self.model)

        self.model.initialize(self.world.trajectory)
        self.history.info["goal_duration"] = self.world.duration

        if TRAJECTORY_CONFIG["traj_type"] == "tracking" and winstor:
            MANAGER_CONFIG["exp_name"] = (
                MANAGER_CONFIG["exp_name"] + f"_trial_{self.world.trial.name}"
            )
            self.setup_paths()

        # Set up plotting
        if MANAGER_CONFIG["live_plot"]:
            self.plotter = Plotter(
                self.frames_folder,
                self.world.trajectory,
                goal_duration=self.world.duration,
            )
            self.plotter.make_figure()

    # ---------------------------------- Set up ---------------------------------- #

    def setup_paths(self):
        if self.winstor:
            main = paths.winstor_main
            self.trials_cache = paths.winstor_trial_cache
        else:
            main = paths.main_fld
            self.trials_cache = paths.trials_cache

        # make main and frames folder
        self.datafolder = main / MANAGER_CONFIG["exp_name"]
        self.datafolder.mkdir(exist_ok=True)
        self.frames_folder = self.datafolder / "frames"
        self.frames_folder.mkdir(exist_ok=True)

    def start_logging(self):
        filename = str(self.datafolder / f"{MANAGER_CONFIG['exp_name']}.log")
        logger.add(filename)
        logger.info(f"Saving data at: {self.datafolder}")

    # ------------------------------ Run simulation ------------------------------ #
    def run(self, n_secs=1):
        # initialize world/history
        # step to frames
        # conclude
        n_steps = int(n_secs / dt)
        print(
            f"\n\n[bold  green]Starting simulation with {n_steps} steps [{n_secs}s at {dt} s/step][/bold  green]"
        )
        with GracefulInterruptHandler() as h:
            with progress:
                task_id = progress.add_task(
                    "running", start=True, total=n_steps
                )

                for itern in range(n_steps):
                    progress.advance(task_id, 1)

                    # Plan
                    curr_x = np.array(self.model.curr_x)
                    g_xs = self.world.plan(curr_x)
                    if g_xs is None:
                        return self.wrap_up()  # DONE !

                    # Solve control
                    u = self.controller.obtain_sol(curr_x, g_xs)

                    # step
                    self.model.step(
                        u, g_xs[1, :]
                    )  # g_xs[1, :] is used to keep a history of state delta

                    # update historu
                    self.history.append(
                        self.model.curr_x,
                        self.model.curr_goal,
                        self.model.curr_wheel_state,
                        self.model.curr_control,
                        dict(trajectory_idx=self.world.curr_traj_waypoint_idx),
                    )

                    # update plot
                    if MANAGER_CONFIG["live_plot"]:
                        self.plotter.update(
                            self.history.record,
                            g_xs,
                            self.world.current_traj_waypoint,
                            itern,
                            elapsed=itern * dt,
                        )

                    # check if we're done
                    if self.world.isdone(self.model.curr_x):
                        return self.wrap_up()

                    if h.interrupted:
                        return self.wrap_up()
        return self.wrap_up()

    # ---------------------------------- Wrap up --------------------------------- #
    def make_video(self):
        try:
            animate_from_images(
                str(self.frames_folder),
                str(self.datafolder / f"{MANAGER_CONFIG['exp_name']}.mp4"),
                int(round(1 / dt)),
            )
        except (ValueError, FileNotFoundError):
            print("Failed to generate video from frames.. ")

        # remove frames folder if everything's okay
        try:
            shutil.rmtree(str(self.frames_folder))
        except (FileNotFoundError, PermissionError):
            print("could not remove frames folder")

    def upload_to_db(self):
        dbx = DropBoxUtils()
        dpx_path = self.datafolder.name
        logger.info(
            f"Uploading data to dropbox at: {dpx_path}", extra={"markup": True}
        )
        try:
            upload_folder(dbx, self.datafolder, dpx_path)
        except Exception as e:
            logger.info(f"Failed to upload to dropbox with error: {e}")
            raise ValueError(f"Failed to upload to dropbox with error: {e}")

    def wrap_up(self):
        # Log config
        for conf in all_configs:
            logger.info(json.dumps(conf, sort_keys=True, indent=4))

        # Save stuff
        self.history.save(
            self.datafolder, self.world.trajectory, self.world.trial
        )

        # make live plot
        if MANAGER_CONFIG["live_plot"]:
            self.make_video()
        else:
            shutil.rmtree(str(self.frames_folder))

        # save summary plot
        plot_results(
            self.datafolder,
            plot_every=15,
            save_path=self.datafolder / "outcome",
        )

        # Upload to dropbox
        if self.winstor:
            try:
                logger.info("Uploading to dropbox")
                self.upload_to_db()
            except Exception as e:
                logger.info(f"Failed to upload to dropbox with error: {e}")
                send_slack_message(
                    f"Failed to upload to dropbox with error: {e}"
                )
            else:
                logger.info("Uploading succesfull")

            shutil.rmtree(str(self.datafolder))
