import datajoint as dj
from loguru import logger
import pandas as pd
from pathlib import Path
import cv2

from fcutils.path import from_yaml, to_yaml, files
from fcutils.progress import track

import sys

sys.path.append("./")
from data.dbase import schema
from data.dbase._tables import (
    insert_entry_in_table,
    print_table_content_to_file,
)

# from data.dbase.io import sort_files
from data.dbase import quality_control as qc
from data.paths import raw_data_folder
from data.dbase import _session, _ccm, _behavior

DO_RECORDINGS_ONLY = True

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
        # fill
        _session.fill_session_table(self)

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

    @staticmethod
    def get_session_tracking_file(session_name):
        tracking_files_folder = raw_data_folder / "tracking"
        video_name = Path(
            (Session & f'name="{session_name}"').fetch1("video_file_path")
        ).stem
        tracking_files = files(
            tracking_files_folder, pattern=f"{video_name}*.h5"
        )

        if not tracking_files:
            logger.warning(f"No tracking data found for {session_name}")
        return tracking_files

    @staticmethod
    def was_tracked(session_name):
        """
            Checks if DLC was ran on the session by looking for a correctly
            named file.
        """
        tracking_files = Session.get_session_tracking_file(session_name)

        if not tracking_files:
            return None
        elif isinstance(tracking_files, Path):
            return True
        else:
            logger.warning(
                f"While looking for tracking data for {session_name} found {len(tracking_files)} tracking files:\n{tracking_files}"
            )
            return True


# ---------------------------------------------------------------------------- #
#                              validated sessions                              #
# ---------------------------------------------------------------------------- #


@schema
class ValidatedSession(dj.Imported):
    definition = """
        # checks that the video and AI files for a session are saved correctly and video/recording are syncd
        -> Session
        ---
        n_frames:                   int  # number of video frames in session
        duration:                   int  # experiment duration ins econds
        n_analog_channels:          int  # number of AI channels recorded in bonsai
        bonsai_cut_start:           int  # where to start/end cutting bonsai signals to align to ephys
        bonsai_cut_end:             int
        ephys_cut_start:            int  # where to start/end cutting bonsai signals to align to bonsai
        ephys_time_scaling_factor:  float  # scales ephys spikes in time to align to bonsai
    """
    analog_sampling_rate = 30000
    excluded_sessions = [
        "FC_210713_AAA1110750_r3_hairpin",  # skipping because cable borke mid recording
    ]

    def make(self, key):
        # fetch data
        session = (Session & key).fetch1()
        has_rec = Session.has_recording(key["name"])
        if has_rec:
            previously_validated_path = "data/dbase/validated_recordings.yaml"
        else:
            previously_validated_path = "data/dbase/validated_sessions.yaml"

        # load previously validated sessions to speed things up
        previously_validated = from_yaml(previously_validated_path)

        # check if session has known problems and should be excluded
        if session["name"] in self.excluded_sessions:
            logger.info(
                f'Skipping session "{session["name"]}" because its in the excluded sessions list'
            )
            return

        # check if validation was already executed on this session
        if session["name"] in previously_validated.keys():
            logger.debug(
                f'Session {session["name"]} was previously validated, loading results'
            )
            key = previously_validated[session["name"]]
        else:
            logger.debug(f'Validating session: {session["name"]}')

            if not has_rec and DO_RECORDINGS_ONLY:
                logger.info(
                    f'Skipping {session["name"]} because we are doing recording sessions ONLY'
                )
                return

            # check bonsai recording was correct
            (
                is_ok,
                analog_nsigs,
                duration_seconds,
                n_frames,
                bonsai_cut_start,
                bonsai_cut_end,
            ) = qc.validate_behavior(
                session["video_file_path"],
                session["ai_file_path"],
                self.analog_sampling_rate,
            )
            if not is_ok:
                logger.warning(f"Session failed to pass validation: {key}")
                return
            else:
                logger.info(f"Session passed BEHAVIOR validation")

            # check ephys data OK and get time scaling factor to align to bonsai
            if has_rec:
                (
                    is_ok,
                    ephys_cut_start,
                    time_scaling_factor,
                ) = qc.validate_recording(
                    session["ai_file_path"],
                    session["ephys_ap_data_path"],
                    sampling_rate=self.analog_sampling_rate,
                )
            else:
                time_scaling_factor, ephys_cut_start = -1, -1

            if not is_ok:
                logger.warning(
                    f"Session failed to pass RECORDING validation: {key}"
                )
                return
            else:
                logger.info(f"Session passed RECORDING validation")

            # prepare data
            key["n_frames"] = int(n_frames)
            key["duration"] = float(duration_seconds)
            key["bonsai_cut_start"] = float(bonsai_cut_start)
            key["bonsai_cut_end"] = float(bonsai_cut_end)
            key["ephys_cut_start"] = float(ephys_cut_start)
            key["ephys_time_scaling_factor"] = float(time_scaling_factor)
            key["n_analog_channels"] = int(analog_nsigs)

            # save results to file
            # if has_rec:
            logger.debug(f"Saving key entries to yaml: {key}")
            previously_validated[session["name"]] = key
            to_yaml(previously_validated_path, previously_validated)

        # fill in table
        logger.info(f'Inserting session data in table: {key["name"]}')
        self.insert1(key)


# ---------------------------------------------------------------------------- #
#                                 behavior data                                #
# ---------------------------------------------------------------------------- #
@schema
class Behavior(dj.Imported):
    definition = """
        # stores AI and csv data from Bonsai in a nicely formatted manner
        -> ValidatedSession
        ---
        speaker:                    longblob  # signal sent to speakers
        pump:                       longblob  # signal sent to pump
        reward_signal:              longblob  # 0 -> 1 when reward is delivered
        reward_available_signal:    longblob  # 1 when the reward becomes available
        trigger_roi:                longblob  # 1 when mouse in trigger ROI
        reward_roi:                 longblob  # 1 when mouse in reward ROI
    """
    analog_sampling_rate = 30000  # in bonsai

    def make(self, key):
        """
            loads data from .bin and .csv data saved by bonsai.

            1. get session
            2. load/cut .bin file from bonsai
            3. load/cut .csv file from bonsai
        """
        # fetch metadata
        name = key["name"]
        try:
            session = (Session * ValidatedSession & f'name="{name}"').fetch1()
        except Exception:
            logger.warning(f"Failed to fetch data for {name} - not validated?")
            return

        # load, format & insert data
        key = _behavior.load_session_data(session, key)
        if key is not None:
            self.insert1(key)


# ---------------------------------------------------------------------------- #
#                           COMMON COORDINATES MATRIX                          #
# ---------------------------------------------------------------------------- #
@schema
class CCM(dj.Imported):
    definition = """
    # stores common coordinates matrix for a session
    -> ValidatedSession
    ---
    correction_matrix:      longblob        # 2x3 Matrix used for correction
    """

    def make(self, key):
        # Get the maze model template

        arena = cv2.imread("data/dbase/arena_template.png")
        arena = cv2.cvtColor(arena, cv2.COLOR_RGB2GRAY)

        # Get path to video
        videopath = (Session & key).fetch1("video_file_path")

        # get matrix
        key["correction_matrix"] = _ccm.get_matrix(videopath, arena)
        self.insert1(key)


# ---------------------------------------------------------------------------- #
#                                 tracking data                                #
# ---------------------------------------------------------------------------- #
@schema
class Tracking(dj.Imported):
    """
        tracking data from DLC. The
        entries in the main table reflext the mouse's body\body axis
        and a sub table is used for each body part.
    
    """

    definition = """
        -> ValidatedSession
        ---
        x:                      longblob  # body position in cm
        y:                      longblob  # body position in cm
        speed:                  longblob  # body speed in cm/s
        orientation:            longblob  # orientation in deg
        angular_velocity:       longblob  # angular velocityi in deg/sec
        direction_of_movement:  longblob  # angle towards where the mouse is moving next
        ---
    """

    class BodyPart(dj.Part):
        definition = """
            -> Tracking
            bpname:  varchar(64)
            ---
            x:                      longblob  # body position in cm
            y:                      longblob  # body position in cm
            speed:                  longblob  # body speed in cm/s
            direction_of_movement:  longblob  # angle towards where the mouse is moving next
        """

    def make(self, key):
        # TODO fill in main and part tables
        raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                                  ephys data                                  #
# ---------------------------------------------------------------------------- #
@schema
class Recording(dj.Imported):
    definition = """
        # stores metadata about the ephys recording
        -> ValidatedSession
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
    # Behavior().drop()
    # sys.exit()

    # -------------------------------- fill dbase -------------------------------- #
    # sort_files()

    logger.info("#####    Filling mouse data")
    # Mouse().fill()

    logger.info("#####    Filling Session")
    # Session().fill()

    logger.info("#####    Validating sessions data")
    # ValidatedSession().populate(display_progress=True)

    logger.info("####     filling CCM")
    # CCM().populate(display_progress=True)

    logger.info("#####    Filling Behavior")
    Behavior().populate(display_progress=True)

    # logger.info('#####    Filling Tracking')
    # Tracking().populate(display_progress=True)

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
