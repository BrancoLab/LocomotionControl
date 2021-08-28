import datajoint as dj
from loguru import logger
import pandas as pd
from pathlib import Path
import cv2
from typing import List, Tuple
import time

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
from data.dbase import (
    _session,
    _ccm,
    _behavior,
    _tracking,
    _probe,
    _triggers,
    _recording,
)
from data.dbase.hairpin_trace import HairpinTrace
from data.dbase.io import get_probe_metadata

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
    def on_hairpin(session_name):
        session = pd.Series((Session & f'name="{session_name}"').fetch1())
        return session.arena == "hairpin"

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
        duration:                   int  # experiment duration in seconds
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


@schema
class BonsaiTriggers(dj.Imported):
    definition = """
        # stores the time (in samples) of camera triggers in bonsai. To registere spikes to them
        -> ValidatedSession
        ---
        trigger_times:   longblob
        n_samples:      int         # tot number of samples in recording
        n_ms:           int         # duration in milliseconds
    """

    def make(self, key):
        session = (Session * ValidatedSession & key).fetch1()
        triggers = _triggers.get_triggers(session)

        key = {**key, **triggers}
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

    likelihood_threshold = 0.99
    cm_per_px = 60 / 830
    bparts = (
        "snout",
        "body",
        "tail_base",
        "left_fl",
        "left_hl",
        "right_fl",
        "right_hl",
    )

    definition = """
        -> ValidatedSession
        ---
        orientation:            longblob  # orientation in deg
        angular_velocity:       longblob  # angular velocityi in deg/sec
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

    class Linearized(dj.Part):
        definition = """
            -> Tracking
            ---
            segment:        longblob  # index of hairpin arena segment
            global_coord:   longblob # values in range 0-1 with position along the arena
        """

    def make(self, key):
        # get tracking data file
        tracking_file = Session.get_session_tracking_file(key["name"])
        if tracking_file is None:
            return

        # get CCM registration matrix
        M = (CCM & key).fetch1("correction_matrix")

        # process data
        key, bparts_keys = _tracking.process_tracking_data(
            key,
            tracking_file,
            M,
            likelihood_th=self.likelihood_threshold,
            cm_per_px=self.cm_per_px,
        )

        self.insert1(key)
        for bpkey in bparts_keys.values():
            self.BodyPart.insert1(bpkey)

        # Get linearized position
        if Session.on_hairpin(key["name"]):
            hp = HairpinTrace()
            key["segment"], key["global_coord"] = hp.assign_tracking(
                bparts_keys["body"]["x"], bparts_keys["body"]["y"]
            )
            del key["orientation"]
            del key["angular_velocity"]
            self.Linearized.insert1(key)

    @staticmethod
    def get_session_tracking(session_name, body_only=True):
        query = Tracking * Tracking.BodyPart & f'name="{session_name}"'

        if body_only:
            query = query & f"bpname='body'"

        if Session.on_hairpin(session_name):
            query = query * Tracking.Linearized

        return pd.DataFrame(query.fetch1())


# ---------------------------------------------------------------------------- #
#                                  ephys data                                  #
# ---------------------------------------------------------------------------- #


@schema
class Probe(dj.Imported):
    definition = """
        # relevant probe information
        -> Mouse
        ---
        skull_coordinates:                              longblob  # AP, ML from bregma in mm
        implanted_depth:                                longblob  # Z axis of stereotax in mm from brain surface
        reconstructed_track_filepath:                   varchar(256)
        angle_ml:                                       longblob
        angle_ap:                                       longblob
    """

    class RecordingSite(dj.Part):
        definition = """
            # metadata about recording sites locations
            -> Probe
            site_id:                        int
            ---
            registered_brain_coordinates:   blob  # in um, in atlas space
            probe_coordinates:              int   # position in um along probe
            brain_region:                   varchar(128)  # acronym
            brain_region_id:                int
            color:                          varchar(128)  # brain region color
        """

    def make(self, key):
        metadata = get_probe_metadata(key["mouse_id"])
        if metadata is None:
            return
        probe_key = {**key, **metadata}

        recording_sites = _probe.place_probe_recording_sites(metadata)
        if recording_sites is None:
            return

        # insert into main table
        self.insert1(probe_key)
        for rsite in recording_sites:
            rsite_key = {**key, **rsite}
            self.RecordingSite.insert1(rsite_key)


@schema
class Recording(dj.Imported):
    definition = """
        # stores metadata about the ephys recording
        -> ValidatedSession
        ---
        concatenated:                       int           # 1 if spike sorting was done on concatenated data
        spike_sorting_params_file_path:     varchar(256)  # PRM file with spike sorting paramters
        spike_sorting_spikes_file_path:     varchar(256)  # CSV files with spikes times
        spike_sorting_clusters_file_path:   varchar(256)  # MAT file with clusters IDs
    """
    recordings_folder = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings"
    )

    def make(self, key):
        # check if the session has a recording
        if not Session.has_recording(key["name"]):
            return

        # load recordings metadata
        rec_metadata = pd.read_excel(
            Session.recordings_metadata_path, engine="odf"
        )

        # Check if it's a concatenated recording
        rec_folder = Path(
            (Session & key).fetch1("ephys_ap_data_path")
        ).parent.parent.name

        concat_filepath = rec_metadata.loc[
            rec_metadata["recording folder"] == rec_folder
        ]["concatenated recording file"].iloc[0]
        if isinstance(concat_filepath, str):
            # it was concatenated
            rec_name = concat_filepath
            rec_path = self.recordings_folder / Path(rec_name)
            key["concatenated"] = 1
            raise NotImplementedError("Check this")
        else:
            rec_name = rec_metadata.loc[
                rec_metadata["recording folder"] == rec_folder
            ]["recording folder"].iloc[0]
            rec_path = (
                self.recordings_folder
                / Path(rec_name)
                / Path(rec_name + "_imec0")
            )
            key["concatenated"] = -1

        # complete the paths to all relevant files
        key["spike_sorting_params_file_path"] = str(
            rec_path / (rec_name + "_t0.imec0.ap.prm")
        )
        key["spike_sorting_spikes_file_path"] = str(
            rec_path / (rec_name + "_t0.imec0.ap.csv")
        )
        key["spike_sorting_clusters_file_path"] = str(
            rec_path / (rec_name + "_t0.imec0.ap_res.mat")
        )

        for name in (
            "spike_sorting_params_file_path",
            "spike_sorting_spikes_file_path",
            "spike_sorting_clusters_file_path",
        ):
            if not Path(key[name]).exists():
                logger.warning(f'Cant file for "{name}"')
                return
        self.insert1(key)


@schema
class Unit(dj.Imported):
    definition = """
        # a single unit's spike sorted data
        -> Recording
        unit_id:        int
        ---
        -> Probe.RecordingSite
        secondary_sites_ids:    longblob  # site_id of each cluster recording site
    """

    class Spikes(dj.Part):
        definition = """
            # spike times in milliseconds and video frame number
            -> Unit
            ---
            spikes_ms:              longblob
            spikes:                 longblob  # in video frames number
        """

    def is_in_target_region(
        unit: dict, targets: List[str]
    ) -> Tuple[bool, bool, str]:
        """
            Checks if any of the unit's recording sites lays into
            target region. Returns True/False based on that, 
            True/False based on if the unit's main site is a target
            and the name of the target region.
        """

        # check if the main unit's site is a target
        main_site = (Probe * Probe.RecordingSite & unit).fetch1()
        if main_site["brain_region"] in targets:
            return True, True, main_site["brain_region"]

        # check secondary sites
        for site_number in unit["secondary_sites_ids"]:
            site = (
                Probe * Probe.RecordingSite & f"site_id={site_number}"
            ).fetch1()
            if site["brain_region"] in targets:
                return True, False, site["brain_region"]

        return False, None, None

    def make_spikes_raster(unit: dict):
        """
            Given a unit's details it makes an array with 0 for every millisecond
            but 1 when the unit spikes
        """
        raise NotImplementedError("Make this happen")

    def make(self, key):
        recording = (Recording & key).fetch1()
        if recording["concatenated"] == 1:
            raise NotImplementedError(
                "Need the adjustment of spike times work for concatenated data"
            )

        # load units data
        units = _recording.load_cluster_curation_results(
            recording["spike_sorting_clusters_file_path"],
            recording["spike_sorting_spikes_file_path"],
        )

        # load behavior camera triggers
        triggers = (BonsaiTriggers & key).fetch1()
        validated = (ValidatedSession & key).fetch1()
        triggers = {**triggers, **validated}

        # fill in units
        for nu, unit in enumerate(units):
            logger.debug(f"     processing unit {nu+1}/{len(units)}")
            # enter info in main table
            unit_key = key.copy()
            unit_key["unit_id"] = unit["unit_id"]
            unit_key["site_id"] = unit["recording_site_id"]

            # get adjusted spike times
            unit_spikes = _recording.get_unit_spike_times(
                unit, triggers, ValidatedSession.analog_sampling_rate
            )
            spikes_key = {**key.copy(), **unit_spikes}
            spikes_key["unit_id"] = unit["unit_id"]

            # insert into table
            self.insert1(unit_key)
            self.Spikes.insert1(spikes_key)

            logger.debug("Inserted unit in table, taking a pause")
            time.sleep(3)


if __name__ == "__main__":
    # ! careful: this is to delete stuff
    # Probe().drop()
    # sys.exit()

    # -------------------------------- fill dbase -------------------------------- #
    # sort_files()

    logger.info("#####    Filling mouse data")
    # Mouse().fill()

    logger.info("#####    Filling Session")
    # Session().fill()

    logger.info("#####    Filling Validated Session")
    # ValidatedSession().populate(display_progress=True)
    # BonsaiTriggers().gipopulate(display_progress=True)

    logger.info("#####    Filling CCM")
    # CCM().populate(display_progress=True)

    logger.info("#####    Filling Behavior")
    # Behavior().populate(display_progress=True)

    logger.info("#####    Filling Tracking")
    # Tracking().populate(display_progress=True)

    logger.info("#####    Filling Probe")
    # Probe().populate(display_progress=True)

    logger.info("#####    Filling Recording")
    # Recording().populate(display_progress=True)

    logger.info("#####    Filling Unit")
    Unit().populate(display_progress=True)

    # ? check probe reconstruction
    # TODO misura l'inserted probe 5 volte e prendi la media
    # TODO crea 5 splines e prendi la media

    # ? unit table
    # TODO: add secondary sites for each unit
    # TODO: check if unit in target regions

    # ? OTHER CHECKS
    # TODO check tracking has the right number of sampels/frames

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
        Probe.RecordingSite,
        Unit,
        Tracking,
        Tracking.BodyPart,
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
        "Tracking",
        "Body Part",
    ]
    for tb, name in zip(TABLES, NAMES):
        print_table_content_to_file(tb, name)
