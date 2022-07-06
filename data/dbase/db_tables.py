try:
    import datajoint as dj
except ImportError:
    have_dj = False
else:
    have_dj = True
from loguru import logger
import pandas as pd
from pathlib import Path
import cv2
from typing import List, Tuple
import numpy as np
import json

from fcutils.path import from_yaml, to_yaml, files
from fcutils.progress import track

import sys

sys.path.append("./")

from data.dbase import schema
from data.dbase._tables import (
    insert_entry_in_table,
    print_table_content_to_file,
)
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
    _locomotion_bouts,
)
from data.dbase.hairpin_trace import HairpinTrace
from data.dbase.io import get_probe_metadata  # , load_bin
from data import data_utils

DO_RECORDINGS_ONLY = False

# ---------------------------------------------------------------------------- #
#                                     mouse                                    #
# ---------------------------------------------------------------------------- #

if have_dj:

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

            for mouse in track(
                data, description="Adding mice", transient=True
            ):
                mouse = mouse["mouse"]

                # add to table
                insert_entry_in_table(
                    mouse["mouse_id"], "mouse_id", mouse, self
                )

    @schema
    class Surgery(dj.Imported):

        definition = """
            # notes when a mouse had a surgery
            -> Mouse
            date:  varchar(256)
            --- 
            type: varchar(256)
            target: varchar(256)
        """

        def make(self, key):
            # see if mouse was implanted with a neuropixel probe
            metadata = get_probe_metadata(key["mouse_id"])
            if metadata is not None:
                key["type"] = "neuropixel"
                key["date"] = metadata["date"]
                key["target"] = metadata["target"]
                self.insert1(key)
                return

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
        too_early_date = 210412  # sessions before this have weird arenas and tracking doesnt do well
        recordings_metadata_path = Path(
            r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\recordings_metadata.ods"
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
            recorded_sessions = recorded_sessions.loc[
                recorded_sessions["USE?"] == "yes"
            ]
            for i, session in recorded_sessions.iterrows():
                if session["bonsai filename"] not in in_table:
                    # raise ValueError(
                    logger.warning(
                        f"Recording session not in table:\n{session}"
                    )

        @staticmethod
        def on_hairpin(session_name):
            session = pd.Series((Session & f'name="{session_name}"').fetch1())
            return session.arena == "hairpin"

        @staticmethod
        def is_too_early(session_name):
            session = pd.Series((Session & f'name="{session_name}"').fetch1())
            return int(session.date) <= Session.too_early_date

        @staticmethod
        def has_recording(session_name):
            """
                Returns True if the session had neuropixel recordings, else False.

                Arguments:
                    session_name: str. Session name
            """
            session = pd.Series((Session & f'name="{session_name}"').fetch1())

            if session.is_recording:
                return True
            else:
                # see perhaps it wes added later
                if (
                    "BAA1101192" in session_name
                    or "BAA0000012" in session_name
                ):
                    session_name = session_name.replace("_hairpin", "")
                    recorded_sessions = pd.read_excel(
                        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\recordings_metadata.ods",
                        engine="odf",
                    )
                    return (
                        session_name
                        in recorded_sessions["bonsai filename"].values
                    )
                else:
                    return False

        @staticmethod
        def get_session_tracking_file(session_name):
            tracking_files_folder = raw_data_folder / "tracking"
            if not tracking_files_folder.exists():
                tracking_files_folder = Path(r"K:\tracking")

            video_name = Path(
                (Session & f'name="{session_name}"').fetch1("video_file_path")
            ).stem
            tracking_files = files(
                tracking_files_folder, pattern=f"{video_name}*.h5"
            )

            if not tracking_files:
                logger.warning(f"No tracking data found for {session_name}")

            elif isinstance(tracking_files, list):
                raise ValueError("Found too many tracking files")
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

    @schema
    class SessionCondition(dj.Imported):
        definition = """
            # stores the conditions (e.g. control, implanted) of an experimental session
            -> Session
            condition:  varchar(256)
        """

        def make(self, key):
            """
                Use Surgery to see if the mouse had surgery by this date and update the key accordingly.
                TODO: add a spreadsheet for nothing special conditions
            """
            session_date = int((Session & key).fetch1("date"))
            # get surgery metadata
            mouse = key["mouse_id"]
            try:
                surgery_date = int(
                    (Surgery & f'mouse_id="{mouse}"').fetch1("date")
                )
                if session_date > surgery_date:
                    key["condition"] = "implanted"
                else:
                    key["condition"] = "naive"
            except:
                key["condition"] = "naive"

            self.insert1(key)

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
            bonsai_cut_start:           int  # start of first bonsai sync pulse
            bonsai_cut_end:             int  # end of last bonsai sync pulse
            ephys_cut_start:            int  # start of first ephys sync pulse
            ephys_cut_end:              int  # end of last ephys sync pulse
            frame_to_drop_pre:          int  # number of frames to drop before the first sync pulse
            frame_to_drop_post:         int  # number of frames to drop after the last sync pulse
            tscale:                     float  # time scaleing factor for ephys
        """
        analog_sampling_rate = 30000
        spikeglx_sampling_rate = 30000.616484

        excluded_sessions = [
            "FC_210713_AAA1110750_r3_hairpin",  # skipping because cable borke mid recording
            "210818_281_longcol_inter_openarena",  # didnt save bonsai data
        ]

        def mark_failed_validation(self, key, reason, nsigs, failed):
            key = (Session & key).fetch1()
            key["__REASON"] = reason
            key["nsigs"] = nsigs
            key["__IS_RECORDING"] = Session.has_recording(key["name"])
            failed[key["name"]] = key
            to_yaml("data/dbase/validation_failed.yaml", failed)

        def make(self, key):
            # check if this session has previously failed validatoin
            failed = from_yaml("data/dbase/validation_failed.yaml")
            if failed is None:
                failed = {}

            if "openarena" in key["name"].lower():
                logger.info(f"Skipping openarena session {key['name']}")
                return

            if key["name"] in failed.keys():
                reason = failed[key["name"]]["__REASON"]
                rec = failed[key["name"]]["__IS_RECORDING"]
                if rec:
                    logger.warning(
                        f'Skipping because "{key["name"]}" (Is recording: {rec}) previously failed validation because: "{reason}"\n\n'
                    )
                return

            # fetch data
            session = (Session & key).fetch1()

            # check if session has known problems and should be excluded
            if session["name"] in self.excluded_sessions:
                logger.info(
                    f'Skipping session "{session["name"]}" because its in the excluded sessions list'
                )
                return
            has_rec = Session.has_recording(key["name"])

            if has_rec:
                # see perhaps it wes added later
                if "BAA1101192" in key["name"] or "BAA0000012" in key["name"]:
                    has_rec = False

            if has_rec:
                previously_validated_path = (
                    "data/dbase/validated_recordings.yaml"
                )
            else:
                previously_validated_path = (
                    "data/dbase/validated_sessions.yaml"
                )

            # load previously validated sessions to speed things up
            previously_validated = from_yaml(previously_validated_path)
            previously_validated = previously_validated or {}

            # check if validation was already executed on this session
            if session["name"] in previously_validated.keys():
                logger.debug(
                    f'Session {session["name"]} was previously validated, loading results'
                )
                key = previously_validated[session["name"]]
                if "frame_to_drop_pre" not in key.keys():
                    key["frame_to_drop_pre"] = 0
                if "frame_to_drop_post" not in key.keys():
                    key["frame_to_drop_post"] = 0
                if "tscale" not in key.keys():
                    key["tscale"] = 1
            else:
                logger.debug(f'Validating session: {session["name"]}')

                # check bonsai recording was correct
                (
                    is_ok,
                    analog_nsigs,
                    duration_seconds,
                    n_frames,
                    bonsai_first_frame,
                    bonsai_last_frame,
                    reason,
                ) = qc.validate_behavior(
                    session["video_file_path"],
                    session["ai_file_path"],
                    self.analog_sampling_rate,
                )
                if not is_ok:
                    self.mark_failed_validation(
                        key, reason, analog_nsigs, failed
                    )
                    logger.warning(
                        f"Session failed to pass BEHAVIOR validation: {key}"
                    )
                    return
                else:
                    logger.info(f"Session passed BEHAVIOR validation")

                # check ephys data OK and get time scaling factor to align to bonsai
                if has_rec:
                    (
                        is_ok,
                        bonsai_cut_start,
                        bonsai_cut_end,
                        ephys_cut_start,
                        ephys_cut_end,
                        frame_to_drop_pre,
                        frame_to_drop_post,
                        tscale,
                        reason,
                    ) = qc.validate_recording(
                        session["ai_file_path"],
                        session["ephys_ap_data_path"],
                        duration_seconds,
                        sampling_rate=self.analog_sampling_rate,
                        ephys_sampling_rate=self.spikeglx_sampling_rate,
                    )
                else:
                    ephys_cut_start, ephys_cut_end = -1, -1
                    bonsai_cut_start, bonsai_cut_end = -1, -1
                    frame_to_drop_pre, frame_to_drop_post = 0, 0
                    tscale = 1

                if not is_ok:
                    logger.warning(
                        f"Session failed to pass RECORDING validation: {key}: {reason}"
                    )
                    self.mark_failed_validation(
                        key, reason, analog_nsigs, failed
                    )
                    return
                elif has_rec:
                    logger.info(f"Session passed RECORDING validation")

                # prepare data
                key["n_frames"] = int(n_frames)
                key["duration"] = float(duration_seconds)
                key["bonsai_cut_start"] = int(bonsai_cut_start)
                key["bonsai_cut_end"] = int(bonsai_cut_end)
                key["ephys_cut_start"] = int(ephys_cut_start)
                key["ephys_cut_end"] = int(ephys_cut_end)
                key["n_analog_channels"] = int(analog_nsigs)
                key["frame_to_drop_pre"] = int(frame_to_drop_pre)
                key["frame_to_drop_post"] = int(frame_to_drop_post)
                key["tscale"] = float(tscale)

                # save results to file
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
            speaker:                    longblob  # signal sent to speakers (signals in frame times)
            pump:                       longblob  # signal sent to pump
            reward_signal:              longblob  # 0 -> 1 when reward is delivered
            reward_available_signal:    longblob  # 1 when the reward becomes available
            trigger_roi:                longblob  # 1 when mouse in trigger ROI
            reward_roi:                 longblob  # 1 when mouse in reward ROI
        """

        def make(self, key):
            """
                loads data from .bin and .csv data saved by bonsai.

                1. get session
                2. load/cut .bin file from bonsai
                3. load/cut .csv file from bonsai
            """
            if DO_RECORDINGS_ONLY and not Session.has_recording(key["name"]):
                logger.debug(
                    f'Skipping {key["name"]} because it is not a recording'
                )
                return

            # fetch metadata
            name = key["name"]
            try:
                session = (
                    Session * ValidatedSession * BonsaiTriggers
                    & f'name="{name}"'
                ).fetch1()
            except Exception:
                logger.warning(
                    f"Failed to fetch data for {name} - not validated?"
                )
                return

            # load, format & insert data
            key = _behavior.load_session_data(
                session, key, ValidatedSession.analog_sampling_rate
            )
            if key is not None:
                self.insert1(key)

    @schema
    class Tones(dj.Computed):
        definition = """
            -> Behavior
            ---
            tone_onsets:                 longblob  # tone onset times in frame number
            tone_offsets:                longblob
        """

        @staticmethod
        def get_session_tone_on(session_name: str) -> np.ndarray:
            n_frames = (ValidatedSession & f'name="{session_name}"').fetch1(
                "n_frames"
            )
            tone_on = np.zeros(n_frames)

            session_tone = (Tones & f'name="{session_name}"').fetch1()
            for on, off in zip(
                session_tone["tone_onsets"], session_tone["tone_offsets"]
            ):
                tone_on[on:off] = 1
            return tone_on

        def make(self, key):
            speaker = (Behavior & key).fetch1("speaker")

            # get tone onsets/offsets times
            try:
                (
                    key["tone_onsets"],
                    key["tone_offsets"],
                ) = data_utils.get_event_times(
                    speaker,
                    kernel_size=211,
                    th=0.005,
                    abs_val=True,
                    debug=False,
                    shift=1,
                )
            except:
                logger.warning(f"Failed to get TONES data for session: {key}")
            else:
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

            # replace with local copy path
            videopath = Path("K:\\vids") / Path(videopath).name
            if not videopath.exists():
                logger.info(f"Could not find video file: {videopath}")

            # get matrix
            key["correction_matrix"] = _ccm.get_matrix(str(videopath), arena)
            self.insert1(key)

    # ---------------------------------------------------------------------------- #
    #                                 tracking data                                #
    # ----------------------------------------------------------------------------
    @schema
    class Tracking(dj.Imported):
        """
            tracking data from DLC. The
            entries in the main table reflect the mouse's body \ body axis
            and a sub table is used for each body part.
        
        """

        likelihood_threshold = 0.95
        cm_per_px = 60 / 830
        bparts = (
            "snout",
            "neck",
            "body",
            "tail_base",
            "left_fl",
            "left_hl",
            "right_fl",
            "right_hl",
            "left_shoulder",
            "left_hip",
            "right_shoulder",
            "right_hip",
        )

        bparts_to_process = (
            "snout",
            "body",
            "neck",
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
            speed:                  longblob  # body speed in cm/sec
            acceleration:           longblob  # in cm/s^2
            u:                      longblob  # longitudingal component of speed (along mouse direction)
            udot:                   longblob  # longitudinal component of acceleration
            theta:                  longblob  # angle towards where the mouse is moving next
            thetadot:               longblob  # rate of change of the direction of movement
            thetadotdot:            longblob # in deg/s^2
        """

        def make(self, key):
            if "_t" in key["name"] and "training" not in key["name"]:
                logger.warning(
                    f'Skipping session {key["name"]} because its a test session'
                )
                return

            if DO_RECORDINGS_ONLY and not Session.has_recording(key["name"]):
                logger.info(
                    f'Skipping {key["name"]} because its not a recording'
                )
                return

            if Session.is_too_early(key["name"]):
                logger.info(
                    f'Skipping session {key["name"]} because its too early - bad tracking'
                )
                return

            # get tracking data file
            tracking_file = Session.get_session_tracking_file(key["name"])
            if tracking_file is None:
                logger.warning("No tracking file found")
                return

            # get number of frames in session
            n_frames = (ValidatedSession & key).fetch1("n_frames")

            # get CCM registration matrix
            M = (CCM & key).fetch1("correction_matrix")

            # process data
            (
                key,
                bparts_keys,
                tracking_n_frames,
            ) = _tracking.process_tracking_data(
                key,
                tracking_file,
                M,
                likelihood_th=self.likelihood_threshold,
                cm_per_px=self.cm_per_px,
                bparts_to_process=self.bparts_to_process,
            )

            # check number of frames
            if n_frames != tracking_n_frames:
                raise ValueError(
                    "Number of frames in video and tracking dont match!!"
                )

            # insert into table
            self.insert1(key)

        @staticmethod
        def get_session_tracking(session_name, body_only=True, movement=True):
            query = Tracking * TrackingBP & f'name="{session_name}"'
            if movement:
                query = query * Movement

            if body_only:
                query = query & f"bpname='body'"

            if Session.on_hairpin(session_name):
                query = query * TrackingLinearized

            if len(query) == 1:
                return pd.DataFrame(query.fetch()).iloc[0]
            else:
                return pd.DataFrame(query.fetch())

    @schema
    class TrackingBP(dj.Imported):
        definition = """
            -> ValidatedSession
            bpname:  varchar(64)
            ---
            x:                          longblob  # body position in cm
            y:                          longblob  # body position in cm
            bp_speed:                   longblob  # body speed in cm/s
            beta:                       longblob   # angle of velocity vector in degrees
        """

        def make(self, key):
            # if key["mouse_id"] not in ("BAA1101192", "BAA0000012"):
            #     return

            # get tracking data file
            tracking_file = Session.get_session_tracking_file(key["name"])
            if tracking_file is None:
                logger.warning("No tracking file found")
                return

            # get CCM registration matrix
            M = (CCM & key).fetch1("correction_matrix")

            # process data
            (
                key,
                bparts_keys,
                tracking_n_frames,
            ) = _tracking.process_tracking_data(
                key,
                tracking_file,
                M,
                likelihood_th=Tracking.likelihood_threshold,
                cm_per_px=Tracking.cm_per_px,
                bparts_to_process=Tracking.bparts_to_process,
            )

            if key is not None:
                for bpkey in bparts_keys.values():
                    self.insert1(bpkey)

    @schema
    class TrackingLinearized(dj.Imported):
        definition = """
            -> Tracking
            ---
            segment:        longblob  # index of hairpin arena segment
            global_coord:   longblob # values in range 0-1 with position along the arena
        """

        def make(self, key):
            # if key["mouse_id"] not in ("BAA1101192", "BAA0000012"):
            #     return

            # Get linearized position
            if Session.on_hairpin(key["name"]):
                body = (TrackingBP & key & "bpname='body'").fetch1()
                hp = HairpinTrace()
                (key["segment"], key["global_coord"],) = hp.assign_tracking(
                    body["x"], body["y"]
                )
                self.insert1(key)

    # ---------------------------------------------------------------------------- #
    #                                  locomotion bouts                            #
    # ---------------------------------------------------------------------------- #
    @schema
    class LocomotionBouts(dj.Imported):
        definition = """
            # identified bouts of continous locomotion
            -> ValidatedSession
            start_frame:        int
            end_frame:          int  # last frame of locomotion bout
            ---
            duration:           float  # duration in seconds
            direction:          varchar(64)   # 'outbound' or 'inbound' or 'none'
            color:              varchar(64)
            complete:           varchar(32)    # True if its form reward to trigger ROIs
            start_roi:          int
            end_roi:            int
            gcoord_delta:       float  # the change in global coordinates during the bout
        """

        speed_th: float = 10  # cm/s
        min_peak_speed = (
            15  # cm/s - each bout must reach this speed at some point
        )
        max_pause: float = 0.5  # (s) if paused for < than this its one contiuous locomotion bout
        min_duration: float = 2  # (s) keep only outs that last at least this long

        min_gcoord_delta: float = 0.25  # the global coordinates must change of at least this during bout

        @staticmethod
        def get_bout_tracking(
            bout: pd.DataFrame, session_tracking: pd.DataFrame = None
        ) -> pd.DataFrame:
            """
                Returns a dictionary with tracking cut to the start/end of the locomotion bout
            """
            session_tracking = (
                Tracking.get_session_tracking(
                    bout["name"], movement=False, body_only=False
                )
                if session_tracking is None
                else session_tracking
            )

            bps = session_tracking.bpname.values
            x = [bp + "_x" for bp in bps]
            y = [bp + "_y" for bp in bps]
            results = {k: [] for k in x + y}
            columns = ("x", "y")
            results["gcoord"] = []

            for bp in bps:
                for col in columns:
                    results[f"{bp}_{col}"] = session_tracking.loc[
                        session_tracking.bpname == bp
                    ][col].iloc[0][bout["start_frame"] : bout["end_frame"]]

            results[f"gcoord"] = session_tracking.loc[
                session_tracking.bpname == "body"
            ]["global_coord"].iloc[0][bout["start_frame"] : bout["end_frame"]]
            return pd.DataFrame(results)

        @staticmethod
        def is_locomoting(session_name: str) -> np.ndarray:
            """
                Returns an array of 1 and 0 with 1 for every frame in which the mouse
                is walking
            """
            n_frames = (ValidatedSession & f'name="{session_name}"').fetch1(
                "n_frames"
            )
            locomoting = np.zeros(n_frames)

            bouts = pd.DataFrame(
                (LocomotionBouts & f'name="{session_name}"').fetch()
            )
            for i, bout in bouts.iterrows():
                locomoting[bout.start_frame : bout.end_frame] = 1
            return locomoting

        @staticmethod
        def get_session_bouts(session_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                (LocomotionBouts & f'name="{session_name}"').fetch()
            )

        def make(self, key):
            if "open" in key["name"]:
                return

            # get tracking data
            tracking = Tracking.get_session_tracking(
                key["name"], body_only=False, movement=False
            )
            if tracking.empty:
                logger.warning(
                    f'Failed to get tracking data for session {key["name"]} - empty df'
                )
                return
            else:
                logger.info(
                    f'Getting locomotion bouts for session {key["name"]}'
                )

            # get bouts
            bouts = _locomotion_bouts.get_session_bouts(
                key,
                tracking,
                Session.on_hairpin(key["name"]),
                speed_th=self.speed_th,
                max_pause=self.max_pause,
                min_duration=self.min_duration,
                min_peak_speed=self.min_peak_speed,
                min_gcoord_delta=self.min_gcoord_delta,
            )

            # insert in table
            for bout in bouts:
                self.insert1(bout)

    @schema
    class ProcessedLocomotionBouts(dj.Manual):
        definition = """
            # processed locomotion bouts exported by Julia (from locomotion bouts from here)
            -> ValidatedSession
            start_frame:        int
            end_frame:          int  # last frame of locomotion bout
            ---
            duration:           float  # duration in seconds
            direction:          varchar(64)   # 'outbound' or 'inbound' or 'none'
            complete:           varchar(32)    # True if its form reward to trigger ROIs
            s:                  longblob  # track position [just body tracking] 
            x:                  longblob  # x position
            y:                  longblob  # y position
            speed:              longblob  # speed in cm/s
            angvel:             longblob  # angular velocity in deg/s
        """
        exported_bouts_folder = Path(
            r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys\locomotion_bouts\processed"
        )

        def fill(self):
            """
                Loads all locomotion bouts and adds them to the table.

                #TODO add part tables for tracking of other body parts
            """

            in_table = self.fetch("name", "start_frame", "end_frame")
            in_table = [f"{n}_{s}_{e}" for n, s, e in zip(*in_table)]

            all_files = [f for f in self.exported_bouts_folder.glob("*.json")]
            for f in track(all_files):
                bout = json.load(open(f))

                key = dict(
                    mouse_id=bout["mouse_id"][0],
                    name=bout["name"][0],
                    start_frame=bout["start_frame"][0],
                    end_frame=bout["end_frame"][0],
                    duration=bout["duration"][0],
                    direction=bout["direction"][0],
                    complete=bout["complete"][0],
                    s=bout["s"],
                    x=bout["x"],
                    y=bout["y"],
                    speed=bout["speed"],
                    angvel=bout["Ï‰"],
                )

                name = f"{key['name']}_{key['start_frame']}_{key['end_frame']}"
                if name not in in_table:
                    self.insert1(key)

    @schema
    class Movement(dj.Imported):
        turning_threshold: float = 20  # deg/sec
        moving_threshold: float = 2.5  # cm/sec

        definition = """
            # stores information about when the mouse is doing certain types of movements
            -> ValidatedSession
            ---
            moving:         longblob  # moving but not necessarily walking
            walking:        longblob  # 1 when the mouse is walking
            turning_left:   longblob
            turning_right:  longblob
            left_fl_moving: longblob
            right_fl_moving: longblob
            left_hl_moving: longblob
            right_hl_moving: longblob
        """

        def make(self, key):
            """
                Gets arrays indicating when the mouse id doing certain kinds of movements
            """
            # get data
            tracking = Tracking.get_session_tracking(
                key["name"], body_only=False, movement=False
            )
            if tracking.empty:
                logger.warning(
                    f'Failed to get tracking data for session {key["name"]}'
                )
                return

            # get when walking
            # key["walking"] = LocomotionBouts.is_locomoting(key["name"])

            # get other movements
            key = _tracking.get_movements(
                key, tracking, self.moving_threshold, self.turning_threshold
            )

            self.insert1(key)

    # ---------------------------------------------------------------------------- #
    #                                  ephys data                                  #
    # ---------------------------------------------------------------------------- #

    @schema
    class Probe(dj.Imported):
        _skip = ["AAA1110751"]
        _tips = {
            "AAA1110750": 400,
        }

        possible_configurations = [
            "b0",
            "longcolumn",
            "r32",
            "r48",
            "r64",
            "r72",
            "r96",
            "r128",
        ]

        definition = """
            # relevant probe information + surgery metadata
            -> Mouse
            ---
            skull_coordinates:                              longblob  # AP, ML from bregma in mm
            implanted_depth:                                longblob  # reconstructed implanted epth in brain in um
            reconstructed_track_filepath:                   varchar(256)
            angle_ml:                                       longblob
            angle_ap:                                       longblob
            target:                                         varchar(128)  # eg "MOs" or "CUN/GRN"
        """

        class RecordingSite(dj.Part):
            definition = """
                # metadata about recording sites locations
                -> Probe
                site_id:                        int
                probe_configuration:            varchar(128)  # b_0, longcol...
                ---
                registered_brain_coordinates:   blob  # in um, in atlas space
                probe_coordinates:              int   # position in um along probe
                brain_region:                   varchar(128)  # acronym
                brain_region_id:                int
                color:                          varchar(128)  # brain region color
            """

        @staticmethod
        def get_session_sites(
            mouse: str, configuration: str = "intref"
        ) -> pd.DataFrame:
            return pd.DataFrame(
                (
                    Probe * Probe.RecordingSite
                    & f'mouse_id="{mouse}"'
                    & f'probe_configuration="{configuration}"'
                ).fetch()
            )

        def make(self, key):
            if key["mouse_id"] in self._skip:
                return

            metadata = get_probe_metadata(key["mouse_id"])
            if metadata is None:
                return

            logger.info(f"Adding Probe for: {key['mouse_id']}")
            probe_key = {**key, **metadata}

            probe_type = "np1" if probe_key["date"] < 220200 else "np24"
            del probe_key["date"]

            # insert into main table
            try:
                self.insert1(probe_key)
            except Exception as e:
                logger.warning(
                    f'Failed to populate probe table, likely missing path to reconstruction file. Original error:\n"{e}"'
                )
                return

            if not probe_key["reconstructed_track_filepath"]:
                #  get files in folder for electrodes placement
                tracks_folder = (
                    Path(
                        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
                    )
                    / probe_key["mouse_id"][-3:]
                )
                if not tracks_folder.exists():
                    logger.warning(
                        f"Could not find tracks folder for {probe_key['mouse_id']}"
                    )
                    return

                probe_key["reconstructed_track_filepath"] = files(
                    tracks_folder
                )

            if probe_type == "np24":
                possible_configurations = self.possible_configurations[2:]
            else:
                possible_configurations = self.possible_configurations[:2]

            tip = 150
            for configuration in possible_configurations:
                logger.info(f"Getting probe configuration: {configuration}")
                recording_sites = _probe.place_probe_recording_sites(
                    probe_key, configuration, tip=tip, probe_type=probe_type
                )
                if recording_sites is None:
                    continue

                for rsite in recording_sites:
                    rsite_key = {**key, **rsite}
                    rsite_key["probe_configuration"] = configuration
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
            recording_probe_configuration:      varchar(256)  # longcol, b_0 ...
            reference:                          varchar(256)  # interf, extref
        """
        # recordings_folder = Path(
        #     r"W:\swc\branco\Federico\Locomotion\raw\recordings"
        # )
        recordings_folder = Path(r"M:\recordings_temp")

        spikeglx_sampling_rate = 30000.616484

        def make(self, key):
            # check if the session has a recording
            if not Session.has_recording(key["name"]):
                logger.info(f"No recording for {key['name']}")
                return

            # load recordings metadata
            rec_metadata = pd.read_excel(
                Session.recordings_metadata_path, engine="odf"
            )

            # get recording folder
            ephis_path = (Session & key).fetch1("ephys_ap_data_path")
            if ephis_path:
                rec_folder = Path(
                    (Session & key).fetch1("ephys_ap_data_path")
                ).parent.parent.name
            else:
                # get it from the recording metadata
                session_name = key["name"].replace("_hairpin", "")
                rec_folder = rec_metadata.loc[
                    rec_metadata["bonsai filename"] == session_name
                ].iloc[0]["recording folder"]

            # get paths and other metadata
            key = _recording.get_recording_filepaths(
                key, rec_metadata, self.recordings_folder, rec_folder
            )
            if key is not None:
                self.insert1(key)

    @schema
    class Unit(dj.Imported):
        precomputed_firing_rate_windows = [10, 25, 100]  # in ms

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

        @staticmethod
        def get_session_units(
            session_name: str,
            probe_configuration: str,
            spikes: bool = False,
            firing_rate: bool = False,
            frate_window: int = 50,
        ) -> pd.DataFrame:

            # query
            query = (
                Unit * Probe.RecordingSite
                & f"name='{session_name}'"
                & f'probe_configuration="{probe_configuration}"'
            )
            if spikes:
                query = query * Unit.Spikes

            # fetch
            units = pd.DataFrame(query)

            # augment
            if firing_rate:
                if frate_window not in Unit.precomputed_firing_rate_windows:
                    raise NotImplementedError()
                else:
                    # load pre-computed firing rates
                    units["firing_rate"] = list(
                        (
                            query * FiringRate & f"bin_width={frate_window}"
                        ).fetch("firing_rate")
                    )
            return units

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
            raise NotImplementedError(
                "this should respect the fact that differentrecordings have different probe configurations"
            )
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

        @staticmethod
        def get_unit_sites(
            mouse: str,
            session_name: str,
            unit_id: int,
            configuration: str = "longcol",
        ) -> pd.DataFrame:
            """
                Gets the recording sites that a unit's spikes are detected on (based on the probe configuration used)
            """
            rsites = Probe.get_session_sites(
                mouse, configuration=configuration
            )
            unit_sites = (
                Unit & f'name="{session_name}"' & f"unit_id={unit_id}"
            ).fetch1("secondary_sites_ids")
            raise NotImplementedError(
                "Double check that this respects probe configurations"
            )

            rsites = rsites.loc[rsites.site_id.isin(unit_sites)]
            return rsites

        def make(self, key):
            logger.info(f'Procesing: "{key["name"]}"')
            recording = (Session * Recording & key).fetch1()

            conf = recording["recording_probe_configuration"]
            rsites = pd.DataFrame(
                (
                    Probe * Probe.RecordingSite
                    & recording
                    & f"probe_configuration='{conf}'"
                ).fetch()
            )

            # load units data
            try:
                units = _recording.load_cluster_curation_results(
                    recording["spike_sorting_clusters_file_path"],
                    recording["spike_sorting_spikes_file_path"],
                )
            except FileNotFoundError as e:
                logger.warning(f"Could not open some file!!!\n{e}")
                return

            # load behavior camera triggers
            triggers = (Session * ValidatedSession & key).fetch1()

            # for M2 recordings manually set the path
            if not triggers["ephys_ap_data_path"]:
                triggers["ephys_ap_data_path"] = (
                    Path("M:\\recordings_temp")
                    / Path(recording["spike_sorting_params_file_path"])
                    .with_suffix(".bin")
                    .name
                )
                if not triggers["ephys_ap_data_path"].exists():
                    logger.warning(
                        f"Could not find the file {triggers['ephys_ap_data_path']}"
                    )
                #     return
                # else:
                triggers["ephys_ap_data_path"] = str(
                    triggers["ephys_ap_data_path"]
                )

            # get time scaling factors
            tscale, ai_file_path = _recording.get_tscale(
                triggers["ephys_ap_data_path"], triggers["ai_file_path"]
            )
            if tscale is None:
                return

            # for M2 triggers get the triggers anew
            if triggers["bonsai_cut_start"] == -1:
                (
                    is_ok,
                    bonsai_cut_start,
                    bonsai_cut_end,
                    ephys_cut_start,
                    ephys_cut_end,
                    frame_to_drop_pre,
                    frame_to_drop_post,
                    tscale,
                    reason,
                ) = qc.validate_recording(
                    ai_file_path,
                    triggers["ephys_ap_data_path"],
                    1,
                    sampling_rate=ValidatedSession.analog_sampling_rate,
                    ephys_sampling_rate=ValidatedSession.spikeglx_sampling_rate,
                    DO_CHECKS=False,
                )
                triggers["bonsai_cut_start"] = bonsai_cut_start
                triggers["bonsai_cut_end"] = bonsai_cut_end
                triggers["ephys_cut_start"] = ephys_cut_start
                triggers["ephys_cut_end"] = ephys_cut_end
                triggers["frame_to_drop_pre"] = frame_to_drop_pre
                triggers["frame_to_drop_post"] = frame_to_drop_post

            # fill in units
            for nu, unit in enumerate(units):
                logger.debug(f"processing unit {nu+1}/{len(units)}")

                # for units "outside" the brain - get the site_id of the closest site
                if (
                    rsites.loc[rsites.site_id == unit["recording_site_id"]]
                    .iloc[0]
                    .brain_region
                    == "OUT"
                ):
                    logger.info("Fixing site_id for unit outside the brain")
                    candidates = rsites.loc[
                        (rsites.site_id.isin(unit["secondary_sites_ids"]))
                        & (rsites.brain_region != "OUT")
                    ]
                    if len(candidates):
                        unit["recording_site_id"] = candidates.iloc[0].site_id
                    else:
                        valid_rsites = rsites.loc[rsites.brain_region != "OUT"]
                        unit_rsite = rsites.loc[
                            rsites.site_id == unit["recording_site_id"]
                        ].iloc[0]
                        selected = np.argmin(
                            (
                                valid_rsites.probe_coordinates.values
                                - unit_rsite.probe_coordinates
                            )
                            ** 2
                        )
                        unit["recording_site_id"] = valid_rsites.iloc[
                            selected
                        ].site_id

                # enter info in main table
                unit_key = key.copy()
                unit_key["unit_id"] = unit["unit_id"]
                unit_key["site_id"] = unit["recording_site_id"]
                unit_key["secondary_sites_ids"] = unit["secondary_sites_ids"]
                unit_key["probe_configuration"] = recording[
                    "recording_probe_configuration"
                ]  # select the right rec site

                # get adjusted spike times
                unit_spikes = _recording.get_unit_spike_times(
                    unit,
                    triggers,
                    ValidatedSession.analog_sampling_rate,
                    tscale=tscale,
                )
                spikes_key = {**key.copy(), **unit_spikes}
                spikes_key["unit_id"] = unit["unit_id"]

                # insert into table
                name = key["mouse_id"]
                (Probe & f'mouse_id="{name}"')
                try:
                    self.insert1(unit_key)
                except Exception as e:
                    raise ValueError(
                        f'Failed to insert key {unit_key} into table\n{Unit()}\nWith error\n"{e}"\nUnit was:{unit}\nProbablyt no Probe key match:\n{Probe & key}'
                    )
                self.Spikes.insert1(spikes_key)

    @schema
    class FiringRate(dj.Imported):
        definition = """
            # spike times in milliseconds and video frame number
            -> Unit
            bin_width:                   float  # std of gaussian kernel in milliseconds
            ---
            firing_rate:                 longblob  # in video frames number
        """

        def make(self, key):
            unit = (Unit * Unit.Spikes * ValidatedSession & key).fetch1()
            logger.info(
                f"Processing: {unit['name']} - unit: {unit['unit_id']}"
            )

            # get firing rates
            for frate_window in Unit.precomputed_firing_rate_windows:
                unit_frate = _recording.get_units_firing_rate(
                    key, unit, frate_window,
                )
                key["firing_rate"] = unit_frate
                key["bin_width"] = frate_window

                self.insert1(key)
                # time.sleep(5)


if __name__ == "__main__":
    # ------------------------------- delete stuff ------------------------------- #
    # ! careful: this is to delete stuff
    # Session().drop()
    # sys.exit()

    # -------------------------------- sorti filex ------------------------------- #

    # logger.info('#####    Sorting FILES')
    # from data.dbase.io import sort_files
    # sort_files()

    # -------------------------------- fill dbase -------------------------------- #

    # TODO check why some recordings are not in the table!!

    logger.info("#####    Filling mouse data")
    # Mouse().fill()

    logger.info("#####    Filling Session")
    # Session().fill()
    # Surgery().populate(display_progress=True)
    # SessionCondition().populate(display_progress=True)

    logger.info("#####    Filling Validated Session")
    # ValidatedSession().populate(display_progress=True)
    # BonsaiTriggers().populate(display_progress=True)

    logger.info("#####    Filling CCM")
    # CCM().populate(display_progress=True)

    logger.info("#####    Filling Behavior")
    # Behavior().populate(display_progress=True)
    # Tones().populate(display_progress=True)

    # ? tracking data
    logger.info("#####    Filling Tracking")
    # Tracking().populate(display_progress=True)
    # TrackingBP().populate(display_progress=True)
    # TrackingLinearized().populate(display_progress=True)
    # LocomotionBouts().populate(display_progress=True)
    # ProcessedLocomotionBouts().fill()
    # Movement().populate(display_progress=True)

    # ? EPHYS
    logger.info("#####    Filling Probe")
    # Probe().populate(display_progress=True)
    # Recording().populate(display_progress=False)

    Unit().populate(display_progress=True)
    # FiringRate().populate(display_progress=True)

    # -------------------------------- print stuff ------------------------------- #
    # print tables contents
    TABLES = [
        Mouse,
        Surgery,
        Session,
        ValidatedSession,
        SessionCondition,
        Probe,
        # Probe.RecordingSite,
        Recording,
        Unit,
        # Movement,
    ]
    NAMES = [
        "Mouse",
        "Surgery",
        "Session",
        "ValidatedSession",
        "SessionCondition",
        "Probe",
        # "RecordingSites",
        "Recording",
        "Unit",
        # "Movement",
    ]
    for tb, name in zip(TABLES, NAMES):
        print_table_content_to_file(tb, name)
