import datajoint as dj
from loguru import logger
import pandas as pd
import numpy as np

from fcutils.path import from_yaml, files
from fcutils.progress import track
from fcutils.maths.signals import get_onset_offset

import sys

sys.path.append("./")
from data.dbase import schema
from data.dbase._tables import sort_files, insert_entry_in_table, load_bin, print_table_content_to_file
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
        video_file_path: varchar(256)
        ai_file_path: varchar(256)
        csv_file_path: varchar(256)
        ephys_ap_data_path: varchar(256)
        ephys_ap_meta_path: varchar(256)
        ephys_lfp_data_path: varchar(256)
        ephys_lfp_meta_path: varchar(256)
    """

    def fill(self):
        raise NotImplementedError('This should also add recording sessions from excel file')
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

            if "test" in name.lower() or "_t" in name.lower():
                logger.warning(
                    f"Skipping session {name} as it is a test recording"
                )
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

    @staticmethod
    def is_validated(session_name):
        raise NotImplementedError


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
class Behavior:
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
            Session * ValidatedSessions & f'name="{session_name}"'
        ).fetch1()
        n_ai_sigs = (ValidatedSessions & f'name="{session_name}"').fetch1(
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
#                                  ephys dataÛ                                 #
# ---------------------------------------------------------------------------- #
@schema
class Recording(dj.Imported):
    definition = """
        # stores metadata about the recording
        -> Sessions
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
        ML_angle:                                       float
        AP_angle:                                       float
        reconstructed_track_file_path_atlas_space:      varchar(256)
        reconstructed_track_file_path_sample_space:     varchar(256)
    """

class RecordingSite(dj.Imported):
    definition = """
        # metadata about recording sites locations
        -> Probe
        id:                             int
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
        -> recording
        id: int
        ---
        -> Probe
        -> RecordingSite
        spike_times: longblob  # spike times registered to the behavior
    """

if __name__ == "__main__":
    # sort files
    sort_files()

    # SessionData.drop()

    # mouse
    # logger.info('#####    Filling mouse data')
    # Mouse().fill()

    # Session
    # logger.info('#####    Filling Session')
    # Session().fill()

    # logger.info('#####    Validating sesions data')
    # ValidatedSessions.populate(display_progress=True)
    # print(ValidatedSessions())

    # logger.info('#####    Filling SessionData')
    SessionData().populate(display_progress=True)
    print(SessionData())


    # print tables contents
    TABLES = [Mouse, Session, ValidatedSession, Behavior, Recording, Probe, RecordingSite, Unit]
    NAMES = ['Mouse', 'Session', 'ValidatedSession', 'Behavior', 'Recording', 'Probe', 'RecordingSite', 'Unit']
    for tb, name in TABLES, NAMES:
        print_table_content_to_file(tb, name)