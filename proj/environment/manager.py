import shutil
import numpy as np
import pandas as pd
import logging
import json

from fcutils.file_io.io import save_yaml

from proj.utils.misc import timestamp
from proj import paths
from proj.animation.animate import animate_from_images
from proj.plotting.results import plot_results
from proj.utils.dropbox import DropBoxUtils, upload_folder
from proj.utils.slack import send_slack_message

from loguru import logger


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

        # create folders
        folders = [self.datafolder, self.frames_folder]
        for fld in folders:
            fld.mkdir(exist_ok=True)

        self._start_logging()

    def _start_logging(self):
        filename = str(self.datafolder / f"{self.exp_name}.log")
        logger.add(filename)
        logger.info("Saving data at: {self.datafolder}")

        # fh = logging.FileHandler(filename)
        # fh.setFormatter(RichHandler(rich_tracebacks=True))
        # log.addHandler(fh)
        # log.addHandler(RichHandler(rich_tracebacks=True))

        # log.info(
        #     f"[bold green] Saving data at: {self.datafolder}",
        #     extra={"markup": True},
        # )

    def _log_conf(self):
        # log config.py
        conf = json.dumps(self.model.config_dict(), sort_keys=True, indent=4)
        logger.info("Config parameters:\n" + conf, extra={"markup": True})

    def _save_results(self):
        # save config
        save_yaml(
            str(self.datafolder / "config.yml"), self.model.config_dict()
        )

        # save trajectory
        np.save(
            str(self.datafolder / "init_trajectory.npy"),
            self.initial_trajectory,
        )

        # save model history
        history = {k: v for k, v in self.model.history.items() if v}
        pd.DataFrame(history).to_hdf(
            str(self.datafolder / "history.h5"), key="hdf"
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
            str(self.datafolder / "cost_history.h5"), key="hdf"
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
        logger.info(
            f"Uploading data to dropbox at: {dpx_path}", extra={"markup": True}
        )

        upload_folder(dbx, self.datafolder, dpx_path)

    def _save_trial(self):
        if self.trial is not None:
            self.trial.to_hdf(str(self.datafolder / "trial.h5"), key="hdf5")

    def conclude(self):
        self._log_conf()
        self._save_results()
        self._save_trial()

        if self.model.LIVE_PLOT:
            self._save_video()

        # save summary plot
        plot_results(
            self.datafolder,
            plot_every=self.plot_every,
            save_path=self.datafolder / "outcome",
        )

        # Upload results to dropbox
        if self.winstor:
            logger.info("Uploading to dropbox", extra={"markup": True})
            try:
                self._upload_to_dropbox()
            except Exception as e:
                logging.error(f"Failed to upload to dropbox: {e}")

            logger.info("Sending slack message", extra={"markup": True})
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
            logger.info("Did not upload to dropbox", extra={"markup": True})

    def failed(self):
        logger.info("Sending slack FAILED message", extra={"markup": True})
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
