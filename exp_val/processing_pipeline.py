import sys
from pathlib import Path
from loguru import logger
from rich.progress import track

from fcutils.video.utils import trim_clip

sys.path.append("./")
from control.utils import from_json, to_json
from exp_val._preprocess_tracking import load_bonsai


class ProcessingPipeline:
    def __init__(self, data_folder):
        """
            Preprocessed a folder of files:
                - load the data and check that everything's okay
                - create cut-out videos of the trials
                - check if tracking data exists, if yes clean it and save it
        """
        fps = 60  # hard coded for now
        trials_clips_fps = 4  # fps of trials clips

        self.data_folder = Path(data_folder)
        self.raw_folder = self.data_folder / "RAW"
        self.tracking_folder = self.data_folder / "TRACKING_DATA"
        self.trials_clips_folder = self.data_folder / "TRIALS_CLIPS"

        # get records
        self.records_path = self.data_folder / "records.json"
        if not self.records_path.exists():
            self.initialize_records()
        self.records = from_json(self.records_path)

        # get experiments name
        self.experiments = self.get_experiments_in_folder()

        # pre-process
        for experiment in track(
            self.experiments, description="preprocessing..."
        ):
            if experiment not in self.records["pre_processed"]:
                # check bonsai saved data correctly
                video_path, stimuli = load_bonsai(self.raw_folder, experiment)

                # make trials clips
                logger.debug(
                    f"Generating clips for {len(stimuli)} trials | fps {fps} -> {trials_clips_fps}"
                )
                for n, stim in enumerate(stimuli):
                    start, end = stim - (2 * fps), stim + (10 * fps)
                    out_vid = self.trials_clips_folder / (
                        f"{experiment}_trial_{n}"
                    )

                    trim_clip(
                        str(video_path),
                        str(out_vid),
                        frame_mode=True,
                        start_frame=start,
                        stop_frame=end,
                        sel_fps=trials_clips_fps,
                    )

                # all done
                self.records["pre_processed"].append(experiment)
        self.update_records()

        # check tracking
        # TODO check for each video if tracking data exists

    def get_experiments_in_folder(self):
        """
            Gets the name of each experiment saved in folder
        """

        def clean(file_name):
            """
                Remove suffixes
            """
            bad = ("_video", "_stimuli", "_camera", "_analog")

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
            f"Preprocess folder found these experiments: {experiments}"
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
        logger.debug(f"Updating records logs: {self.records}")
        to_json(self.records, self.records_path)


if __name__ == "__main__":
    data_folder = "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD_raw"
    ProcessingPipeline(data_folder)
