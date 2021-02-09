from loguru import logger
import numpy as np
import json
from rich import print
import shutil
from pathlib import Path

import sys


from fcutils.progress import progress
from fcutils.path import from_json

from pyrnn._utils import GracefulInterruptHandler
from pyinspect.utils import timestamp

from control._io import DropBoxUtils, upload_folder
from control.live_plot import Plotter
from control.plot import plot_results
from control.history import History

from control import config
from control import paths

from control.world import World
from control.control import Controller
from control.model import Model


class Manager:
    def __init__(
        self,
        winstor=False,
        trialn=None,
        config_file=None,
        folder=None,
        to_db=True,
        trajectory_file=None,
    ):
        """
            Main class to run a control simulation and saving the results

            Arguments:
                winstor: bool. Set to true when running remotely on HPC
                trialn: int. Set to int to use a specific trial when running on tracking data
                config_file: str, Path. Path to a config .json file to replace parameters set
                    in control/config.py (e.g. for hyperparameters optimization)
                folder: str, Path. Path to folder where data will be saved (override default)
                to_db: bool. If true data will be uploaded to dropbox when done
                trajectory_file: str, Path. Path to a .npy file with trajectory data.
        """
        self.to_db = to_db
        self.winstor = winstor

        # Set up
        # setup experiment name
        if config.TRAJECTORY_CONFIG["traj_type"] == "tracking" and winstor:
            config.MANAGER_CONFIG["exp_name"] = (
                config.MANAGER_CONFIG["exp_name"] + f"_trial_{trialn}"
            )
        else:
            config.MANAGER_CONFIG["exp_name"] = (
                config.MANAGER_CONFIG["exp_name"]
                + f"_{timestamp()}_{np.random.randint(10000)}"
            )
        self.setup_paths(folder=folder)

        # set up params
        if config_file is not None:
            self.override_configs(config_file)

        # start logger
        self.start_logging()

        # Set up classes
        self.history = History()
        self.world = World(
            self.trials_cache, trialn, trajectory=trajectory_file
        )
        self.model = Model()
        self.controller = Controller(self.model)

        self.model.initialize(self.world.trajectory)
        self.history.info["goal_duration"] = self.world.duration

        # Set up plotting
        if config.MANAGER_CONFIG["live_plot"]:
            self.plotter = Plotter(
                self.frames_folder,
                self.world.trajectory,
                goal_duration=self.world.duration,
            )
            self.plotter.make_figure()

    # ---------------------------------- Set up ---------------------------------- #

    def override_configs(self, config_file):
        """
            Replace parameters from control/config.py with those
            specified in a config .json file. 
        """
        logger.debug(f"Setting new params from file: {config_file}")

        # load
        new_params = from_json(config_file)

        # keep experiment name
        new_params["MANAGER_CONFIG"]["exp_name"] = config.MANAGER_CONFIG[
            "exp_name"
        ]

        # fix matrices
        new_params["CONTROL_CONFIG"] = {
            k: v if isinstance(v, int) else np.diag(v)
            for k, v in new_params["CONTROL_CONFIG"].items()
        }

        # override
        config.CONTROL_CONFIG = new_params["CONTROL_CONFIG"]
        config.MANAGER_CONFIG = new_params["MANAGER_CONFIG"]
        config.MOUSE = new_params["MOUSE"]
        config.PARAMS = new_params["PARAMS"]
        config.PLANNING_CONFIG = new_params["PLANNING_CONFIG"]
        config.TRAJECTORY_CONFIG = new_params["TRAJECTORY_CONFIG"]
        config.iLQR_CONFIG = new_params["iLQR_CONFIG"]

        config.all_configs = (
            config.MANAGER_CONFIG,
            config.TRAJECTORY_CONFIG,
            config.MOUSE,
            config.PLANNING_CONFIG,
            config.iLQR_CONFIG,
            {k: str(v) for k, v in config.CONTROL_CONFIG.items()},
            config.PARAMS,
        )

    def setup_paths(self, folder=None):
        if self.winstor:
            main = paths.winstor_main
            self.trials_cache = paths.winstor_trial_cache
        else:
            main = paths.main_fld
            self.trials_cache = paths.trials_cache

        # override default save folder
        if folder is None:
            self.datafolder = main / config.MANAGER_CONFIG["exp_name"]
        else:
            self.datafolder = Path(folder)

        # make main and frames folder
        self.datafolder.mkdir(exist_ok=True)
        self.frames_folder = self.datafolder / "frames"
        self.frames_folder.mkdir(exist_ok=True)

    def start_logging(self):
        # logger.warning("Soimulation specific logging disabled")
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        filename = str(self.datafolder / "log.log")
        logger.add(filename, level="DEBUG")
        logger.info(f"Saving data at: {self.datafolder}")

        # Log config
        for conf in config.all_configs:
            logger.info(json.dumps(conf, sort_keys=True, indent=4))

    # ------------------------------ Run simulation ------------------------------ #
    def run(self, n_secs=1):
        # initialize world/history
        # step to frames
        # conclude
        n_steps = int(n_secs / config.PARAMS["dt"])
        logger.info(
            f"\n\n[bold  green]Starting simulation with {n_steps} steps [{n_secs}s at {config.PARAMS['dt']} s/step][/bold  green]"
        )
        with GracefulInterruptHandler() as h:
            with progress:
                task_id = progress.add_task(
                    "running", start=True, total=n_steps
                )

                for itern in range(n_steps):
                    if itern % 100 == 0:
                        logger.info(f"Iteration: {itern}")
                    self.itern = itern
                    progress.advance(task_id, 1)

                    # change params for first few steps
                    if itern < 5:
                        is_warmup = True
                        logger.debug(f"Step: {itern} | warmup phase")
                        config.CONTROL_CONFIG["R"] = config.CONTROL_CONFIG[
                            "R_start"
                        ]
                        config.CONTROL_CONFIG["Z"] = config.CONTROL_CONFIG[
                            "Z_start"
                        ]
                        config.PLANNING_CONFIG[
                            "prediction_length"
                        ] = config.PLANNING_CONFIG["prediction_length_start"]
                    else:
                        is_warmup = False
                        config.CONTROL_CONFIG["R"] = config.CONTROL_CONFIG[
                            "R_run"
                        ]

                        config.PLANNING_CONFIG[
                            "prediction_length"
                        ] = config.PLANNING_CONFIG["prediction_length_run"]

                        # change Z a bit later
                        if itern > 7:
                            config.CONTROL_CONFIG["Z"] = config.CONTROL_CONFIG[
                                "Z_run"
                            ]

                    # Plan
                    curr_state = np.array(self.model.curr_x)
                    goal_states = self.world.plan(curr_state)
                    if goal_states is None:
                        logger.info(
                            "It looks like we are done, world.plan did not return any goal states"
                        )
                        return self.wrap_up()  # DONE !

                    # Solve control
                    u = self.controller.solve(curr_state, goal_states)

                    # step
                    self.model.step(
                        u, goal_states[1, :]
                    )  # goal_states[1, :] is used to keep a history of state delta

                    # update historu
                    self.history.append(
                        self.model.curr_x,
                        self.model.curr_goal,
                        self.model.curr_control,
                        self.model.curr_wheel_velocities,
                        dict(trajectory_idx=self.world.curr_traj_waypoint_idx),
                    )

                    # update plot
                    if config.MANAGER_CONFIG["live_plot"]:
                        self.plotter.update(
                            self.world.xy_original,
                            self.history.record,
                            goal_states,
                            self.world.current_traj_waypoint,
                            itern,
                            elapsed=itern * config.PARAMS["dt"],
                            is_warmup=is_warmup,
                        )

                    # check if we're done
                    if self.world.isdone(self.model.curr_x):
                        logger.info(
                            "world says we are done because we are close to the end goal"
                        )
                        return self.wrap_up()

                    if h.interrupted:
                        logger.info("Manual interruption, exiting")
                        return self.wrap_up()

        logger.info(f"Simulation finished after {itern} frames")
        return self.wrap_up()

    # ---------------------------------- Wrap up --------------------------------- #
    def make_video(self):
        # remove frames folder if everything's okay
        try:
            shutil.rmtree(str(self.frames_folder))
        except (FileNotFoundError, PermissionError, OSError):
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
        logger.info("Wrapping up simulation")

        # Save stuff
        self.history.save(
            self.datafolder, self.world.trajectory, self.world.trial
        )

        # make live plot
        if config.MANAGER_CONFIG["live_plot"]:
            self.make_video()
        else:
            shutil.rmtree(str(self.frames_folder))

        # save summary plot
        logger.info("Generating plots")
        plot_results(
            self.datafolder,
            plot_every=15,
            save_path=self.datafolder / "outcome",
        )

        # Upload to dropbox
        if self.winstor and self.to_db:
            try:
                logger.info("Uploading to dropbox")
                self.upload_to_db()
            except Exception as e:
                logger.info(f"Failed to upload to dropbox with error: {e}")
            else:
                logger.info("Uploading succesfull")
                shutil.rmtree(str(self.datafolder))
