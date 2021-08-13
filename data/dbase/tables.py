import datajoint as dj
from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path

from fcutils.path import from_yaml, files
from fcutils.progress import track
from fcutils.maths.signals import get_onset_offset

import sys

sys.path.append("./")
from data.dbase import schema
from data.dbase._tables import (
    insert_entry_in_table,
    print_table_content_to_file,
)
from data.dbase.io import load_bin, sort_files
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
        # a session is one experiment on one day on one mouse | this keeps track of the paths
        -> Mouse
        name: varchar(128)
        ---
        date: varchar(256)
        is_recording: int   # 1 for recordings and 0 else
        arena: varchar(64)
        video_file_path: varchar(256)
        ai_file_path: varchar(256)
        csv_file_path: varchar(256)
        ephys_ap_data_path: varchar(256)
        ephys_ap_meta_path: varchar(256)
    """

    recordings_metadata_path = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings_metadata.ods"
    )
    recordings_raw_data_path = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings"
    )

    def fill(self):
        logger.info("Filling in session table")
        in_table = Session.fetch("name")
        mice = from_yaml("data\dbase\mice.yaml")

        # Load recordings sessoins metadata
        recorded_sessions = pd.read_excel(
            self.recordings_metadata_path, engine="odf"
        )

        # Get the videos of all sessions
        vids = [
            f for f in files(raw_data_folder / "video") if ".avi" in f.name
        ]

        for video in track(
            vids, description="Adding sessions", transient=True
        ):
            # Get session data
            name = video.name.split("_video")[0]
            if name in in_table:
                continue

            if "test" in name.lower() in name.lower():
                logger.warning(f"Skipping session {name} as it is a test")
                continue

            # get date and mouse
            try:
                date = name.split("_")[1]
                mouse = [
                    m["mouse"]["mouse_id"]
                    for m in mice
                    if m["mouse"]["mouse_id"] in name
                ][0]
            except IndexError:
                logger.warning(
                    f"Skipping session {name} because couldnt figure out the mouse or date it was done on"
                )
                continue
            key = dict(mouse_id=mouse, name=name, date=date)

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

            # get ephys files & arena type
            if name in recorded_sessions["bonsai filename"].values:
                rec = recorded_sessions.loc[
                    recorded_sessions["bonsai filename"] == name
                ].iloc[0]
                base_path = (
                    self.recordings_raw_data_path
                    / rec["recording folder"]
                    / (rec["recording folder"] + "_imec0")
                    / (rec["recording folder"] + "_t0.imec0")
                )
                key["ephys_ap_data_path"] = str(base_path) + ".ap.bin"
                key["ephys_ap_meta_path"] = str(base_path) + ".ap.meta"

                key["arena"] = rec.arena
                key["is_recording"] = 1
                key["date"] = rec.date
            else:
                key["ephys_ap_data_path"] = ""
                key["ephys_ap_meta_path"] = ""
                key["arena"] = "hairpin"
                key["is_recording"] = 0

            # add to table
            insert_entry_in_table(key["name"], "name", key, self)

        # check everything went okay
        self.check_recordings_complete()

    def check_recordings_complete(self):
        """
            Checks that all recording sessions are in the table
        """
        in_table = Session.fetch("name")
        recorded_sessions = pd.read_excel(
            self.recordings_metadata_path, engine="odf"
        )
        for i, session in recorded_sessions.iterrows():
            if session["bonsai filename"] not in in_table:
                raise ValueError(f"Recording session not in table:\n{session}")

        if len((Session & "is_recording=1").fetch()) != len(recorded_sessions):
            raise ValueError(
                "Not enough recorded sessions in table, but not sure which one is missing"
            )

    @staticmethod
    def has_recording(session_name):
        """
            Returns True if the session had neuropixel recordings, else False.

            Arguments:
                session_name: str. Session name
        """
        session = pd.Series((Session & f'name="{session_name}"').fetch1())
        return session.is_recording


# ---------------------------------------------------------------------------- #
#                              validated sessions                              #
# ---------------------------------------------------------------------------- #


@schema
class ValidatedSession(dj.Imported):
    definition = """
        # checks that the video and AI files for a session are saved correctly and video/recording are syncd
        -> Session
        ---
        n_analog_channels: int  # number of AI channels recorded in bonsai
        bonsai_cut_start: int  # where to start/end cutting bonsai signals to align to ephys
        bonsai_cut_end: int
        ephys_cut_start: int
        ephys_cut_end: int
        ephys_time_scaling_factor: float  # scales ephys spikes in time to align to bonsai
    """
    analog_sampling_rate = 30000

    def make(self, key):
        session = (Session & key).fetch1()
        logger.debug(f'Validating session: {session["name"]}')

        # check bonsai recording was correct
        is_ok, analog_nsigs = qc.validate_bonsai(
            session["video_file_path"],
            session["ai_file_path"],
            self.analog_sampling_rate,
        )

        if Session.has_recording(key["name"]):
            (
                b_cut_start,
                b_cut_end,
                e_cut_start,
                e_cut_end,
            ) = qc.validate_recording(
                session["ai_file_path"],
                session["ephys_ap_data_path"],
                sampling_rate=self.analog_sampling_rate,
            )
        else:
            b_cut_start, b_cut_end, e_cut_start, e_cut_end = -1, -1, -1, -1

        key["bonsai_cut_start"] = b_cut_start
        key["bonsai_cut_end"] = b_cut_end
        key["ephys_cut_start"] = e_cut_start
        key["ephys_cut_end"] = e_cut_end

        if is_ok:
            # all OK, add to table
            key["n_analog_channels"] = analog_nsigs
            self.insert1(key)


# ---------------------------------------------------------------------------- #
#                                 behavior data                                #
# ---------------------------------------------------------------------------- #
@schema
class Behavior(dj.Imported):
    definition = """
        # stores AI and csv data in a nicely formatted manner
        -> Session
        ---
        speaker:                    longblob  # signal sent to speakers
        pump:                       longblob  # signal sent to pump
        reward_signal:              longblob  # 0 -> 1 when reward is delivered
        reward_available_signal:    longblob  # 1 when the reward becomes available
        trigger_roi:                longblob  # 1 when mouse in trigger ROI
        reward_roi:                 longblob  # 1 when mouse in reward ROI
        duration:                   float     # session duration in seconds
    """
    analog_sampling_rate = 30000  # in bonsai

    def get(self, session_name):
        """
            loads data from .bin and .csv data saved by bonsai.

            1. get session
            2. load/cut .bin file from bonsai
            3. load/cut .csv file from bonsai
        """

        session = (
            Session * ValidatedSession & f'name="{session_name}"'
        ).fetch1()
        n_ai_sigs = (ValidatedSession & f'name="{session_name}"').fetch1(
            "n_analog_channels"
        )
        logger.debug(f'Making SessionData for session: {session["name"]}')

        # load analog
        analog = load_bin(session["ai_file_path"], nsigs=n_ai_sigs)

        # get start and end frame times
        frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
        session["duration"] = (
            frame_trigger_times[-1] - frame_trigger_times[0]
        ) / self.analog_sampling_rate

        # get analog inputs between frames start/end times
        _analog = analog[frame_trigger_times[0] : frame_trigger_times[-1]] / 5
        session["pump"] = 5 - _analog[:, 1]  # 5 -  to invert signal
        session["speaker"] = _analog[:, 2]

        # get camera frame and probe syn times
        session["frame_trigger_times"] = (
            get_onset_offset(analog[:, 0], 2.5)[0] - frame_trigger_times[0]
        )

        if Session.has_recording(session["name"]):
            session["probe_sync_trigger_times"] = (
                get_onset_offset(analog[:, 3], 2.5)[0] - frame_trigger_times[0]
            )
        else:
            session["probe_sync_trigger_times"] = -1

        # load csv data
        logger.debug("Loading CSV")
        data = pd.read_csv(session["csv_file_path"])
        if len(data.columns) < 5:
            logger.warning("Skipping because of incomplete CSV")
            return None  # first couple recordings didn't save all data

        data.columns = [
            "ROI activity",
            "lick ROI activity",
            "mouse in ROI",
            "mouse in lick ROI",
            "deliver reward signal",
            "reward available signal",
        ]

        # make sure csv data has same length as the number of frames (off by max 2)
        delta = len(frame_trigger_times) - len(data)
        if delta > 2:
            raise ValueError(
                f"We got {len(frame_trigger_times)} frames but CSV data has {len(data)} rows"
            )
        if not delta:
            raise NotImplementedError("This case is not covered")
        pad = np.zeros(delta)

        # add key entries
        session["reward_signal"] = np.concatenate(
            [data["deliver reward signal"].values, pad]
        )
        session["trigger_roi"] = np.concatenate(
            [data["mouse in ROI"].values, pad]
        )
        session["reward_roi"] = np.concatenate(
            [data["mouse in lick ROI"].values, pad]
        )
        session["reward_available_signal"] = np.concatenate(
            [data["reward available signal"].values, pad]
        )

        return session


# ---------------------------------------------------------------------------- #
#                                  ephys data                                  #
# ---------------------------------------------------------------------------- #
@schema
class Recording(dj.Imported):
    definition = """
        # stores metadata about the recording
        -> Session
        ---
        probe_file_path:                    varchar(256)
        spike_sorting_params_file_path:     varchar(256)
        spike_sorting_spikes_file_path:     varchar(256)
        spike_sorting_clusters_file_path:   varchar(256)
    """


@schema
class Probe(dj.Imported):
    definition = """
        # relevant probe information
        -> Recording
        ---
        skull_coordinates:                              longblob  # AP, ML from bregma in mm
        implanted_depth:                                longblob  # Z axis of stereotax in mm
        reconstructed_track_file_path_atlas_space:      varchar(256)
        reconstructed_track_file_path_sample_space:     varchar(256)
        angle_ml:                                        longblob
        angle_ap:                                        longblob
    """


@schema
class RecordingSite(dj.Imported):
    definition = """
        # metadata about recording sites locations
        -> Probe
        site_id:                             int
        ---
        probe_coords:                   blob
        brain_coordinates:              blob  # in sample space
        registered_brain_coordinates:   blob  # in atlas space
        brain_region:                   blob

    """


@schema
class Unit(dj.Imported):
    definition = """
        # a single unit's spike sorted data
        -> Recording
        unit_id: int
        ---
        -> RecordingSite
        site_id: int
        spike_times: longblob  # spike times registered to the behavior
    """


if __name__ == "__main__":
    # ! careful: this is to delete stuff
    # Session().drop()
    # sys.exit()

    # -------------------------------- fill dbase -------------------------------- #
    # sort files
    sort_files()

    # mouse
    logger.info("#####    Filling mouse data")
    Mouse().fill()

    # Session
    logger.info("#####    Filling Session")
    Session().fill()

    # logger.info('#####    Validating sesions data')
    # ValidatedSessions.populate(display_progress=True)
    # print(ValidatedSessions())

    # logger.info('#####    Filling Behavior')
    # Behavior().populate(display_progress=True)
    # print(Behavior())

    # -------------------------------- print stuff ------------------------------- #
    # print tables contents
    TABLES = [
        Mouse,
        Session,
        pd.DataFrame((Session & "is_recording=1").fetch()),
        ValidatedSession,
        Behavior,
        Recording,
        Probe,
        RecordingSite,
        Unit,
    ]
    NAMES = [
        "Mouse",
        "Session",
        "Recordings",
        "ValidatedSession",
        "Behavior",
        "Recording",
        "Probe",
        "RecordingSite",
        "Unit",
    ]
    for tb, name in zip(TABLES, NAMES):
        print_table_content_to_file(tb, name)
