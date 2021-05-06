import datajoint as dj
from loguru import logger
import pandas as pd

from fcutils.path import from_yaml
from fcutils.progress import track
from fcutils.maths.signals import get_onset_offset

import sys

sys.path.append("./")
from data.dbase import schema
from data.dbase._tables import sort_files, insert_entry_in_table, load_bin
from data.paths import raw_data_folder
from data.dbase import quality_control as qc

# ---------------------------------------------------------------------------- #
#                                     mouse                                    #
# ---------------------------------------------------------------------------- #


@schema
class Mouse(dj.Manual):
    definition = """
        # represents mice
        mouse_id: varchar(128)
        ---
        strain: varchar(64)
        dob: varchar(64)
    """

    def fill(self):
        """
            fills in the table
        """
        data = from_yaml("experimental_validation\hairpin\dbase\mice.yaml")
        logger.info("Filling in mice table")

        for mouse in track(data, description="Adding mice", transient=True):
            mouse = mouse["mouse"]

            # add to table
            insert_entry_in_table(mouse["mouse_id"], "mouse_id", mouse, self)


# ---------------------------------------------------------------------------- #
#                                   sessions                                   #
# ---------------------------------------------------------------------------- #


@schema
class Session(dj.Manual):
    definition = """
        # a session is one experiment on one day on one mouse
        -> Mouse
        name: varchar(128)
        ---
        training_day: int
        video_file_path: varchar(256)
        ai_file_path: varchar(256)
        csv_file_path: varchar(256)
        ephys_ap_data_path: varchar(256)
        ephys_ap_meta_path: varchar(256)
        ephys_lfp_data_path: varchar(256)
        ephys_lfp_meta_path: varchar(256)
    """

    def fill(self):
        raise NotImplementedError(
            "No need to add metadata manually, just check which files are there"
        )
        data = from_yaml("experimental_validation\hairpin\dbase\sessions.yaml")
        logger.info("Filling in session table")

        for session in track(
            data, description="Adding sessions", transient=True
        ):
            key = dict(mouse_id=session["mouse"], name=session["name"])

            # get file paths
            key["video_file_path"] = (
                raw_data_folder / "video" / (session["name"] + "_video.avi")
            )
            key["ai_file_path"] = (
                raw_data_folder
                / "analog_inputs"
                / (session["name"] + "_analog.bin")
            )

            key["csv_file_path"] = (
                raw_data_folder
                / "analog_inputs"
                / (session["name"] + "_data.csv")
            )

            if (
                not key["video_file_path"].exists()
                or not key["ai_file_path"].exists()
            ):
                raise FileNotFoundError(
                    f"Either video or AI files not found for session: {session}"
                )

            # get training day
            key["training_day"] = session["training_day"]

            # add to table
            insert_entry_in_table(key["name"], "name", key, self)

            if session["ephys"]:
                raise NotImplementedError

    @staticmethod
    def has_recording(session_name):
        """
            Returns True if the session had neuropixel recordings, else False.

            Arguments:
                session_name: str. Session name
        """
        session = pd.Series(
            (Session & f'session_name="{session_name}"').fetch1()
        )
        if len(session.ephys_ap_data_path):
            return True
        else:
            return False


# ---------------------------------------------------------------------------- #
#                              validated sessions                              #
# ---------------------------------------------------------------------------- #


@schema
class ValidatedSessions(dj.Imported):
    definition = """
        # checks that the video and AI files for a session are saved correctly and video/recording are syncd
        -> Session
    """
    analog_sampling_rate = 30000

    def make(self, key):
        session = (Session & key).fetch1()
        logger.debug(f'Validating session: {session["name"]}')

        # check bonsai recording was correct
        is_ok = qc.validate_bonsai(
            session["video_file_path"],
            session["ai_file_path"],
            self.analog_sampling_rate,
        )

        if Session.has_recording(key["session_name"]):
            is_ok = qc.validate_recording(
                session["ai_file_path"], session["ephys_ap_data_path"]
            )

        if is_ok:
            # all OK, add to table to avoid running again in the future
            self.insert1(key)


@schema
class SessionData(dj.Imported):
    definition = """
        # stores AI and csv data in a nicely formatted manner
        -> ValidatedSessions
        ---
        speaker: longblob
        pump: longblob
        roi_activity: longblob
        mouse_in_roi: longblob
        reward_signal: longblob
        duration: float  # duration in seconds
        lick_roi_activity: longblob
        frame_triggers: longblob
        probe_sync: longblob
    """
    analog_sampling_rate = 30000  # in bonsai

    def make(self, key):
        session = (Session & key).fetch1()
        raise NotImplementedError
        logger.debug(f'Loading SessionData for session: {session["name"]}')

        # load analog
        logger.debug("Loading analog")
        analog = load_bin(session["ai_file_path"], nsigs=4)

        # get start and end frame times
        frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
        key["duration"] = (
            frame_trigger_times[-1] - frame_trigger_times[0]
        ) / self.analog_sampling_rate

        # get cut analog inputs
        key["speaker"] = (
            analog[frame_trigger_times[0] : frame_trigger_times[-1], 2]
        ) / 5
        key["pump"] = (
            5 - analog[frame_trigger_times[0] : frame_trigger_times[-1], 1]
        ) / 5  # 5 -  to invert signal

        # load csv data
        logger.debug("Loading CSV")
        data = pd.read_csv(session["csv_file_path"])
        data.columns = [
            "ROI activity",
            "lick ROI activity",
            "mouse in ROI",
            "mouse in lick ROI",
            "deliver reward signal",
            "reward available signal",
        ]
        # cut csv data between frames -- CSV is already saved only when a frame is acquired

        # save in table


if __name__ == "__main__":
    sort_files()

    # # mouse
    # logger.info('#####    Filling mouse data')
    # Mouse().fill()

    # # Session
    # # Session.drop()

    # logger.info('#####    Filling Session')
    # Session().fill()

    # logger.info('#####    Validating sesions data')
    # ValidatedSessions.populate(display_progress=True)

    # logger.info('#####    Filling SessionData')
    # SessionData().populate(display_progress=True)
