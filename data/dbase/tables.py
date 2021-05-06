import datajoint as dj
from loguru import logger
import pandas as pd

from fcutils.path import from_yaml
from fcutils.progress import track

import sys

sys.path.append("./")
from data.dbase import schema
from data.dbase._tables import sort_files, insert_entry_in_table
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
        -> Session:
        speaker_signal: longblob
        pump: longblob
        roi_activity: longblob
        mouse_in_roi: longblob
        reward_signal: longblob
        lick_roi_activity: longblob
        frame_triggers: longblob
        probe_sync: longblob
    """
    analog_sampling_rate = 30000  # in bonsai

    def make(self, key):
        session = (Session & key).fetch1()
        print(session)

        # load analog data
        # get first and last frame

        # cut analog data between frames

        # load csv data
        # cut csv data between frames

        # save in table


if __name__ == "__main__":
    sort_files()

    Mouse().fill()

    Session().fill()
    ValidatedSessions.populate(display_progress=True)
