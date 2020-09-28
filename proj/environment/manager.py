import shutil
import numpy as np
import pandas as pd
from fancylog import fancylog
import logging

from rich.logging import RichHandler

from fcutils.file_io.io import save_yaml


from proj.utils.misc import timestamp
from proj import paths
from proj.animation.animate import animate_from_images
from proj.plotting.results import plot_results
from proj.utils.dropbox import DropBoxUtils, upload_folder
from proj.utils.slack import send_slack_message

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class Manager:
    def __init__(self, model, winstor=False):
        """
            This class manages paths, saving of results etc.
        """
        self.simstart = timestamp()

        self.model = model
        self.exp_name = f'{model.SIMULATION_NAME}_{model.trajectory["name"]}_{timestamp()}_{np.random.randint(low=0, high=10000)}'

        # get main folder
        if winstor:
            # ? winstor
            main_fld = paths.winstor_main
            self.datafolder = main_fld / self.exp_name

            self.trials_cache = paths.winstor_trial_cache

        else:
            main_fld = paths.main_fld
            self.datafolder = main_fld / "cache"
            self.trials_cache = paths.trials_cache

            # clean cache
            try:
                shutil.rmtree(str(self.datafolder))
            except (FileNotFoundError, PermissionError):
                pass

        # get subfolders
        self.frames_folder = self.datafolder / "frames"
        self.results_folder = self.datafolder / "results"

        # create folders
        folders = [self.datafolder, self.frames_folder, self.results_folder]
        for fld in folders:
            fld.mkdir(exist_ok=True)

        self._start_logging()

    def _start_logging(self):
        # Start logging
        fancylog.start_logging(
            output_dir=str(self.datafolder),
            filename=self.exp_name + ".log",
            multiprocessing_aware=False,
            write_git=False,
            verbose=False,
            write_cli_args=False,
            file_log_level="INFO",
        )

        # log main folder
        log = logging.getLogger("rich")
        log.setLevel(logging.INFO)

        log.info(
            f"[bold green] Saving data at: {self.datafolder}",
            extra={"markup": True},
        )

    def _log_conf(self):
        # log config.py
        try:
            with open("proj/model/config.py") as f:
                conf = "\n" + f.read()
        except FileNotFoundError:
            conf = self.model.config_dict()
        logging.info(conf)

    def _save_results(self):
        # save config
        save_yaml(
            str(self.results_folder / "config.yml"), self.model.config_dict()
        )

        # save trajectory
        np.save(
            str(self.results_folder / "init_trajectory.npy"),
            self.initial_trajectory,
        )

        # save model history
        history = {k: v for k, v in self.model.history.items() if v}
        pd.DataFrame(history).to_hdf(
            str(self.results_folder / "history.h5"), key="hdf"
        )

        # save the last frame as a results image
        try:
            last_frame = [
                f for f in self.frames_folder.glob("*.png") if f.is_file()
            ][-1]
            shutil.copy(
                str(last_frame), str(self.datafolder / "final_frame.png")
            )
        except IndexError:
            pass  # no frames were saved

        # save cost history
        pd.DataFrame(self.cost_history).to_hdf(
            str(self.results_folder / "cost_history.h5"), key="hdf"
        )

    def _save_video(self):
        # make gif
        try:
            animate_from_images(
                str(self.frames_folder),
                str(self.datafolder / f"{self.exp_name}.mp4"),
                10,
            )
        except (ValueError, FileNotFoundError):
            print("Failed to generate video from frames.. ")
        else:
            # remove frames folder if everything's okay
            try:
                shutil.rmtree(str(self.frames_folder))
            except (FileNotFoundError, PermissionError):
                print("could not remove frames folder")

    def _upload_to_dropbox(self):
        dbx = DropBoxUtils()
        dpx_path = self.datafolder.name
        logging.info(f"Uploading data to dropbox at: {dpx_path}")

        upload_folder(dbx, self.datafolder, dpx_path)

    def conclude(self):
        self._log_conf()
        self._save_results()

        if self.model.LIVE_PLOT:
            self._save_video()

        # save summary plot
        plot_results(
            self.results_folder,
            plot_every=self.plot_every,
            save_path=self.datafolder / "outcome",
        )

        # Upload results to dropbox
        if self.winstor:
            logging.info("Uploading to dropbox")
            try:
                self._upload_to_dropbox()
            except Exception as e:
                logging.error(f"Failed to upload to dropbox: {e}")

            logging.info("Sending slack message")
            send_slack_message(
                f"""
                \n
                Completed simulation
                Start time: {self.simstart}
                End time: {timestamp()}
                Data folder: {self.datafolder}
                """
            )
        else:
            logging.info("Did not upload to dropbox")

    def failed(self):
        logging.info("Sending slack FAILED message")
        if self.winstor:
            send_slack_message(
                f"""
                \n
                Failed simulation simulation
                Start time: {self.simstart}
                End time: {timestamp()}
                Data folder: {self.datafolder}
                """
            )
