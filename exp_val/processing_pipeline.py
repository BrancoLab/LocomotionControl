import sys
from pathlib import Path
from loguru import logger
from rich.progress import track

from fcutils.video.utils import trim_clip
from pyinspect.utils import dir_files

sys.path.append("./")
from control.utils import from_json, to_json
from exp_val._preprocess_tracking import load_bonsai, make_bash_text


class ProcessingPipeline:
    def __init__(self, data_folder):
        """
            Preprocessed a folder of files:
                - load the data and check that everything's okay
                - create cut-out videos of the trials
                - check if tracking data exists, if yes clean it and save it
        """
        self.fps = 60  # hard coded for now
        self.trials_clips_fps = 4  # fps of trials clips

        self.data_folder = Path(data_folder)
        self.raw_folder = self.data_folder / "RAW"
        self.tracking_folder = self.data_folder / "TRACKING_DATA"
        self.tracking_bash_folder = self.data_folder / "tracking_bash_scripts"
        self.trials_clips_folder = self.data_folder / "TRIALS_CLIPS"

        # get records
        self.records_path = self.data_folder / "records.json"
        if not self.records_path.exists():
            self.initialize_records()
        self.records = from_json(self.records_path)

        # get experiments name
        self.experiments = self.get_experiments_in_folder()

        # pre-process
        self.preprocess()

        # check tracking
        self.check_tracking()

    def get_experiments_in_folder(self):
        """
            Gets the name of each experiment saved in folder
        """

        def clean(file_name):
            """
                Remove suffixes
            """
            bad = ("_video", "_stimuli", "_camera", "_analog", "_orig")

            for bd in bad:
                file_name = file_name.replace(bd, "")

            return file_name

        # Get each unique experiment name
        experiments = set(
            [
                clean(f.stem)
                for f in self.raw_folder.glob("FC_*")
                if f.is_file() and "test" not in f.name
            ]
        )

        logger.debug(
            f"Preprocess folder found these {len(experiments)} experiments"
        )
        return experiments

    def initialize_records(self):
        """
            Creates a json file to keep track of what's processed

        """
        content = dict(pre_processed=[], tracked=[],)
        to_json(content, self.records_path)

    def update_records(self):
        """
            Saves the records to file
        """
        to_json(self.records, self.records_path)

    def preprocess(self):
        """
            Checks that bonsai saved all fine correctly and loads some necessary stuff
        """
        for experiment in track(
            self.experiments, description="preprocessing..."
        ):
            if experiment not in self.records["pre_processed"]:
                logger.debug(f"\n\n\n[b magenta]Preprocessing: {experiment}")
                # check bonsai saved data correctly
                video_path, stimuli = load_bonsai(
                    self.raw_folder, experiment, self.fps
                )

                # make trials clips
                logger.debug(
                    f"Generating clips for {len(stimuli)} trials | fps {self.fps} -> {self.trials_clips_fps}"
                )
                for n, stim in enumerate(stimuli):
                    start, end = stim - (2 * self.fps), stim + (10 * self.fps)
                    out_vid = self.trials_clips_folder / (
                        f"{experiment}_trial_{n}.mp4"
                    )

                    trim_clip(
                        str(video_path),
                        str(out_vid),
                        frame_mode=True,
                        start_frame=start,
                        stop_frame=end,
                        sel_fps=self.trials_clips_fps,
                    )

                # all done for this experiment
                self.records["pre_processed"].append(experiment)
                self.update_records()

    def check_tracking(self):
        """
            Checks that all videos have been tracked with dlc, 
            if not it creates a .sh file for each video
            so that it can be tracked on HPC
        """
        # check which experiments have been tracked
        tracked = [
            f.name.split("_videoDLC")[0]
            for f in dir_files(self.tracking_folder, "*.h5")
        ]
        for exp in self.experiments:
            if exp in tracked and exp not in self.records["tracked"]:
                self.records["tracked"].append(exp)
                self.update_records()

        # create bash files for experiments to track
        for experiment in track(
            self.experiments, description="checking tracking..."
        ):
            bash_path = self.tracking_bash_folder / f"{experiment}.sh"
            if experiment not in self.records["tracked"]:
                # generate a bash script for tracking
                logger.debug(
                    f"Generating tracking bash script for: {experiment}"
                )

                with open(bash_path, "w") as out:
                    out.write(
                        make_bash_text(
                            experiment,
                            str(self.raw_folder / f"{experiment}_video.avi"),
                            str(self.tracking_folder),
                        )
                    )
            else:
                # already tracked, remove bash script
                if bash_path.exists():
                    bash_path.unlink()

        logger.info(
            f"{len(dir_files(self.tracking_bash_folder))} videos left to track with DLC"
        )


if __name__ == "__main__":
    data_folder = "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD"
    ProcessingPipeline(data_folder)
