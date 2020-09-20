import shutil
import numpy as np
import pandas as pd
from fancylog import fancylog
import logging
from rich.logging import RichHandler

from fcutils.file_io.io import save_yaml


from proj.utils import timestamp
from proj import paths
from proj.animation.animate import animate_from_images
from proj.plotting.results import plot_results

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class Manager:
    def __init__(self, model, winstor=False):
        """
            This class manages paths, saving of results etc.
        """
        self.model = model

        self.exp_name = f'{model.trajectory["name"]}_{timestamp()}_{np.random.randint(low=0, high=10000)}'

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

        # Start logging
        fancylog.start_logging(
            output_dir=str(self.datafolder),
            filename=self.exp_name + ".txt",
            multiprocessing_aware=False,
            write_git=False,
            verbose=False,
            write_cli_args=False,
            file_log_level="INFO",
        )

        print(f"Saving data at: {self.datafolder}")
        logging.info("\n\n\n\n")
        logging.info(f"Saving data at: {self.datafolder}")

        log = logging.getLogger("rich")
        log.info(
            f"[bold green] Saving data at: {self.datafolder}",
            extra={"markup": True},
        )

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
        last_frame = [
            f for f in self.frames_folder.glob("*.png") if f.is_file()
        ][-1]
        shutil.copy(str(last_frame), str(self.datafolder / "final_frame.png"))

        # save cost history
        pd.DataFrame(self.cost_history).to_hdf(
            str(self.results_folder / "cost_history.h5"), key="hdf"
        )

    def _save_video(self):
        # make gif
        try:
            fps = int(np.ceil(1 / self.model.dt))
            animate_from_images(
                str(self.frames_folder),
                str(self.datafolder / f"{self.exp_name}.mp4"),
                fps,
            )
        except (ValueError, FileNotFoundError):
            print("Failed to generate video from frames.. ")
        else:
            # remove frames folder if everything's okay
            try:
                shutil.rmtree(str(self.frames_folder))
            except (FileNotFoundError, PermissionError):
                print("could not remove frames folder")

    def conclude(self):
        self._save_results()
        self._save_video()

        # save summary plot
        plot_results(
            self.results_folder,
            plot_every=self.plot_every,
            save_path=self.datafolder / "outcome",
        )
