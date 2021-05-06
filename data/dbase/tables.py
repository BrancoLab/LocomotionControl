import datajoint as dj
from loguru import logger
import pandas as pd

from fcutils.path import from_yaml, files
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
        data = from_yaml("data\dbase\mice.yaml")
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
        logger.info("Filling in session table")
        in_table = Session.fetch("name")

        # Get the videos of all sessions
        vids = [
            f for f in files(raw_data_folder / "video") if ".avi" in f.name
        ]

        # Get all ephys sessions names
        rec_files = files(raw_data_folder / "recordings")
        if rec_files is not None:
            ephys_files = [f for f in rec_files if ".ap.bin" in f]
            ephys_sessions = [f.name.split("_g0")[0] for f in ephys_files]
            raise NotImplementedError("Need to debug this part")
        else:
            ephys_sessions = []

        for video in track(
            vids, description="Adding sessions", transient=True
        ):
            # Get session data
            name = video.name.split("_video")[0]
            if name in in_table:
                continue

            if "test" in name.lower():
                continue

            try:
                _, date, mouse, day = name.split("_")
            except ValueError:
                logger.warning(
                    f"Skipping session {name} - likely a test recording"
                )
                continue

            key = dict(mouse_id=mouse, name=name, training_day=int(day[1:]))

            # get file paths
            key["video_file_path"] = (
                raw_data_folder / "video" / (name + "_video.avi")
            )
            key["ai_file_path"] = (
                raw_data_folder / "analog_inputs" / (name + "_analog.bin")
            )

            key["csv_file_path"] = (
                raw_data_folder / "analog_inputs" / (name + "_data.csv")
            )

            if (
                not key["video_file_path"].exists()
                or not key["ai_file_path"].exists()
            ):
                raise FileNotFoundError(
                    f"Either video or AI files not found for session: {name} with data:\n{key}"
                )

            # Get ephys files
            if name in ephys_sessions:
                logger.debug(f"Session {name} has ephys recordings")
                key["ephys_ap_data_path"] = (
                    raw_data_folder
                    / "recordings"
                    / f"{name}_g0_t0.imec0.ap.bin"
                )

                key["ephys_ap_meta_path"] = (
                    raw_data_folder
                    / "recordings"
                    / f"{name}_g0_t0.imec0.ap.meta"
                )

                key["ephys_lfp_data_path"] = (
                    raw_data_folder
                    / "recordings"
                    / f"{name}_g0_t0.imec0.lf.bin"
                )

                key["ephys_lfp_meta_path"] = (
                    raw_data_folder
                    / "recordings"
                    / f"{name}_g0_t0.imec0.lf.meta"
                )

            else:
                key["ephys_ap_data_path"] = ""
                key["ephys_ap_meta_path"] = ""
                key["ephys_lfp_data_path"] = ""
                key["ephys_lfp_meta_path"] = ""

            # add to table
            insert_entry_in_table(key["name"], "name", key, self)

        print(Session())

    @staticmethod
    def has_recording(session_name):
        """
            Returns True if the session had neuropixel recordings, else False.

            Arguments:
                session_name: str. Session name
        """
        session = pd.Series((Session & f'name="{session_name}"').fetch1())
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

        if Session.has_recording(key["name"]):
            is_ok = qc.validate_recording(
                session["ai_file_path"], session["ephys_ap_data_path"]
            )

        if is_ok:
            # all OK, add to table
            self.insert1(key)


@schema
class SessionData(dj.Imported):
    definition = """
        # stores AI and csv data in a nicely formatted manner
        -> ValidatedSessions
        ---
        speaker:                    longblob  # signal sent to speakers
        pump:                       longblob  # signal sent to pump
        reward_signal:              longblob  # 0 -> 1 when reward is delivered
        trigger_roi:                longblob  # 1 when mouse in trigger ROI
        reward_roi:                 longblob  # 1 when mouse in reward ROI
        duration:                   float     # session duration in seconds
        frame_trigger_times:        longblob  # when frame triggers are sent, in samples
        probe_sync_trigger_times:   longblob  # when probe sync triggers are sent, in samples
    """
    analog_sampling_rate = 30000  # in bonsai

    def make(self, key):
        """
            loads data from .bin and .csv data saved by bonsai.

            1. get session
            2. load/cut .bin file from bonsai
            3. load/cut .csv file from bonsai

            # TODO deal with sessoins without probe sync trigger times
        """
        raise NotImplementedError
        session = (Session & key).fetch1()
        logger.debug(f'Loading SessionData for session: {session["name"]}')

        # load analog
        analog = load_bin(session["ai_file_path"], nsigs=4)

        # get start and end frame times
        frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
        key["duration"] = (
            frame_trigger_times[-1] - frame_trigger_times[0]
        ) / self.analog_sampling_rate

        # get analog inputs between frames start/end times
        _analog = analog[frame_trigger_times[0] : frame_trigger_times[-1]] / 5
        key["frames_triggers"]
        key["pump"] = 5 - _analog[:, 1]  # 5 -  to invert signal
        key["speaker"] = _analog[:, 2]

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

    # mouse
    # logger.info('#####    Filling mouse data')
    # Mouse().fill()

    # Session
    # logger.info('#####    Filling Session')
    # Session().fill()

    # logger.info('#####    Validating sesions data')
    ValidatedSessions.populate(display_progress=True)

    # logger.info('#####    Filling SessionData')
    # SessionData().populate(display_progress=True)
