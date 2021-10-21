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
    _opto,
    _roi,
)
from data.dbase.hairpin_trace import HairpinTrace
from data.dbase.io import get_probe_metadata, get_opto_metadata, load_bin
from data import data_utils
from data import arena

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
        opto_surgery_metadata_file = Path(
            r"W:\swc\branco\Federico\Locomotion\raw\opto_surgery_metadata.ods"
        )
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
                key['type'] = 'neuropixel'
                key['date'] = metadata['date']
                key['target'] = metadata['target']
                self.insert1(key)
                return

            # see if the mouse was implanted with optic cannula
            logger.warning('Implement Surgery for Opto')
            # metadata = get_opto_metadata(
            #     key["mouse_id"], self.opto_surgery_metadata_file
            # )
            # if metadata is not None:
            #     key['type'] = 'optogenetics'
            #     key['date'] = metadata['date']
            #     key['target'] = metadata['target']
            #     self.insert1(key)
            #     return


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
            recorded_sessions = recorded_sessions.loc[
                recorded_sessions["USE?"] == "yes"
            ]
            for i, session in recorded_sessions.iterrows():
                if session["bonsai filename"] not in in_table:
                    raise ValueError(
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
            # stores the conditions (e.g. control, implanted) of an experimental sessoin
            -> Session
            condition:  varchar(256)
        """

        def make(self, key):
            '''
                Use Surgery to see if the mouse had surgery by this date and update the key accordingly.
                TODO: add a spreadsheet for nothing special conditions
            '''
            session_date = int((Session & key).fetch1('date'))
            # get surgery metadata
            mouse = key['mouse_id']
            try:
                surgery_date = int((Surgery & f'mouse_id="{mouse}"').fetch1('date'))
                if session_date > surgery_date:
                    key['condition'] = 'implanted'
                else:
                    key['condition'] = 'naive'
            except:
                key['condition'] = 'naive'
        
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
            bonsai_cut_start:           int  # where to start/end cutting bonsai signals to align to ephys
            bonsai_cut_end:             int
            ephys_cut_start:            int  # where to start/end cutting bonsai signals to align to bonsai
            ephys_time_scaling_factor:  float  # scales ephys spikes in time to align to bonsai
        """
        analog_sampling_rate = 30000
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
            if key["name"] in failed.keys():
                logger.warning(
                    f'Skipping because {key["name"]} previously failed validation'
                )
                return

            # fetch data
            session = (Session & key).fetch1()
            has_rec = Session.has_recording(key["name"])
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
                        ephys_cut_start,
                        time_scaling_factor,
                        reason,
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
                    self.mark_failed_validation(
                        key, reason, analog_nsigs, failed
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

            # get matrix
            key["correction_matrix"] = _ccm.get_matrix(videopath, arena)
            self.insert1(key)

    # ---------------------------------------------------------------------------- #
    #                                 tracking data                                #
    # ----------------------------------------------------------------------------
    @schema
    class Tracking(dj.Imported):
        """
            tracking data from DLC. The
            entries in the main table reflect the mouse's body\body axis
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
        )

        definition = """
            -> ValidatedSession
            ---
            orientation:            longblob  # orientation in deg
            angular_velocity:       longblob  # angular velocityi in deg/sec
            speed:                  longblob  # body speed in cm/sec
            acceleration:           longblob  # in cm/s^2
            theta:                  longblob  # angle towards where the mouse is moving next
            thetadot:               longblob  # rate of change of the direction of movement
            thetadotdot:            longblob # in deg/s^2
        """

        class BodyPart(dj.Part):
            definition = """
                -> Tracking
                bpname:  varchar(64)
                ---
                x:                          longblob  # body position in cm
                y:                          longblob  # body position in cm
                bp_speed:                   longblob  # body speed in cm/s
            """

        class Linearized(dj.Part):
            definition = """
                -> Tracking
                ---
                segment:        longblob  # index of hairpin arena segment
                global_coord:   longblob # values in range 0-1 with position along the arena
            """

        def make(self, key):
            _key = key.copy()

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
            )

            # check number of frames
            if n_frames != tracking_n_frames:
                raise ValueError(
                    "Number of frames in video and tracking dont match!!"
                )

            # insert into table
            self.insert1(key)
            for bpkey in bparts_keys.values():
                self.BodyPart.insert1(bpkey)

            # Get linearized position
            if Session.on_hairpin(key["name"]):
                hp = HairpinTrace()
                lin_key = _key.copy()
                (
                    lin_key["segment"],
                    lin_key["global_coord"],
                ) = hp.assign_tracking(
                    bparts_keys["body"]["x"], bparts_keys["body"]["y"]
                )
                self.Linearized.insert1(lin_key)

        @staticmethod
        def get_session_tracking(session_name, body_only=True, movement=True):
            query = Tracking * Tracking.BodyPart & f'name="{session_name}"'
            if movement:
                query = query * Movement

            if body_only:
                query = query & f"bpname='body'"

            if Session.on_hairpin(session_name):
                query = query * Tracking.Linearized

            if len(query) == 1:
                return pd.DataFrame(query.fetch()).iloc[0]
            else:
                return pd.DataFrame(query.fetch())

    # ---------------------------------------------------------------------------- #
    #                                  locomotion bouts                            #
    # ---------------------------------------------------------------------------- #
    @schema
    class ROICrossing(dj.Imported):
        min_speed = 20  # cm/s, only roi enters with this speed are considered
        max_duration = 8   # roi crossing must last <= this

        definition = """
            # when the mouse enters a ROI
            -> ValidatedSession
            -> SessionCondition
            roi:  varchar(64)
            start_frame:    int
            end_frame:      int
            crossing_id:    int
            ---
            mouse_exits: int  # 1 if the mouse exists the ROI in time
            duration: float
            gcoord:             longblob
            x:                  longblob
            y:                  longblob
            speed:              longblob
            acceleration:       longblob
            theta:              longblob
            thetadot:           longblob
            thetadotdot:        longblob
        """

        class InitialCondition(dj.Part):
            definition = """
                # stores the conditions at the time of enter
                -> ROICrossing
                ---
                x_init:               float
                y_init:               float
                speed_init:           float
                acceleration_init:    float
                theta_init:           float
                thetadot_init:        float
            """

        def make(self, key):
            if not Session.on_hairpin(key["name"]):
                return 
                
            # get tracking data
            tracking = Tracking.get_session_tracking(
                key["name"], body_only=True, movement=False
            )
            if tracking.empty:
                return
            else:
                logger.info(
                    f'Getting ROI crossings for session {key["name"]}'
                )

            # get bouts
            crossings = []
            for ROI in arena.ROIs:
                crossings.extend(_roi.get_rois_crossings(
                        tracking,
                        ROI,
                        int(self.max_duration * 60),
                        min_speed=self.min_speed,
                    )
            )

            # insert in table
            for cross in crossings:
                cross['crossing_id'] = len(self) + 1
                self.insert1({**key, **cross})

            # insert in part table
            for cross in crossings:
                part_key = key.copy()
                for k in ('roi', 'start_frame', 'end_frame', 'crossing_id'):
                    part_key[k] = cross[k]

                for k in ('x', 'y', 'speed', 'acceleration', 'theta', 'thetadot'):
                    part_key[k+'_init'] = cross[k][0]
                self.InitialCondition.insert1(part_key)

    @schema
    class RoiCrossingsTwins(dj.Imported):
        definition = """
            # for each roi crossing, find a twin crossing with same initial conditions
            -> ROICrossing
            twin_id:  int  # ID of ROICrossing thin to the selected one.
        """
        def make(self, key):
            # get all crossings from the same session
            session = key['name']
            crossings = pd.DataFrame(ROICrossing * ROICrossing.InitialCondition & f'name="{session}"')

            key['twin_id'] = _roi.select_twin_crossing(crossings, key['crossing_id'])
            if key['twin_id']:
                self.insert1(key)
            

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
            if DO_RECORDINGS_ONLY and not Session.has_recording(key["name"]):
                logger.debug(
                    f'Skipping {key["name"]} because it doesnt have a recording'
                )
                return

            # get tracking data
            tracking = Tracking.get_session_tracking(
                key["name"], body_only=False, movement=False
            )
            if tracking.empty:
                logger.warning(
                    f'Failed to get tracking data for session {key["name"]}'
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
        """

        def make(self, key):
            """
                Gets arrays indicating when the mouse id doing certain kinds of movements
            """
            # get data
            tracking = Tracking.get_session_tracking(key["name"])

            # get when walking
            key["walking"] = LocomotionBouts.is_locomoting(key["name"])

            # get other movements
            key = _tracking.get_movements(
                key, tracking, self.moving_threshold, self.turning_threshold
            )

            self.insert1(key)

    # ---------------------------------------------------------------------------- #
    #                                 OPTOGENETICS                                 #
    # ---------------------------------------------------------------------------- #

    @schema
    class OptoImplant(dj.Imported):
        definition = """
            # metadata about opto experiment surgeries
            -> Mouse
            ---
            skull_coordinates:                              longblob  # AP, ML from bregma in mm
            implanted_depth:                                longblob  # Z axis of stereotax in um from brain surface
            injected_depth:                                 longblob  # Z axis of injection
            virus_1:                                        varchar(128)  # name of virus
            virus_2:                                        varchar(128)  # name of second virus
            injection_volume:                               int  # in nL
            target:                                         varchar(128)  # eg "MOs" or "CUN/GRN"
        """

        opto_surgery_metadata_file = Path(
            r"W:\swc\branco\Federico\Locomotion\raw\opto_surgery_metadata.ods"
        )

        def make(self, key):
            metadata = get_opto_metadata(
                key["mouse_id"], self.opto_surgery_metadata_file
            )
            if metadata is None:
                return
            else:
                key = {**key, **metadata}
                self.insert1(key)

    @schema
    class OptoSession(dj.Manual):
        definition = """
            # metadata about experiments with OPTO stimulation
            -> ValidatedSession
            ---
            roi_1:                int  # 1 if used, 0 otherwise
            roi_2:                int  # 1 if used, 0 otherwise
            roi_3:                int  # 1 if used, 0 otherwise
            roi_4:                int  # 1 if used, 0 otherwise
            roi_5:                int  # 1 if used, 0 otherwise
        """

        opto_session_metadata_file = Path(
            r"W:\swc\branco\Federico\Locomotion\raw\opto_metadata.ods"
        )

        def fill(self):
            _opto.fill_opto_table(self, Session)

    @schema
    class OptoStimuli(dj.Imported):
        definition = """
            # collects the time stamps of each laser stimulation
            -> OptoSession
            ---
            stim_onsets:            longblob  # stimuli start times in frame number
            stim_offsets:           longblob  # stimuli start times in frame number
            stim_roi:               longblob  # ROI number, based on tracking data
            stim_power:             longblob  # stim power in mW # TODO make conversion factor
        """

        def make(self, key):
            # get AI of opto stim from bin file
            session = (Session * ValidatedSession & key).fetch1()
            opto_signal = load_bin(
                session["ai_file_path"], nsig=session["n_analog_channels"]
            )[:, -1]

            logger.warning(
                "OptoStimuli does not currently extract STIM_ROI info from tracking"
            )

            # extract stim times
            try:
                (
                    key["stim_onsets"],
                    key["stim_offsets"],
                ) = data_utils.get_event_times(
                    opto_signal,
                    kernel_size=211,
                    th=0.005,
                    abs_val=False,
                    debug=True,
                    shift=1,
                )

                key["stim_ROI"] = np.ones(
                    len(key["stim_onsets"])
                )  # ! this needs implementing
            except:
                logger.warning(f"Failed to get TONES data for session: {key}")
            else:
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
        possible_configurations = ["b0", "longcolumn"]
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
            probe_key = {**key, **metadata}
            del probe_key['date']

            logger.info(
                f'\n\================    Getting reconstructed probe position for mouse {key["mouse_id"]}'
            )

            # insert into main table
            self.insert1(probe_key)

            # get recording sites in each possible configuration
            tip = (
                self._tips[key["mouse_id"]]
                if key["mouse_id"] in self._tips.keys()
                else 175
            )
            for configuration in self.possible_configurations:
                recording_sites = _probe.place_probe_recording_sites(
                    metadata, configuration, tip=tip
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
            recording_probe_configuration:                varchar(256)  # longcol, b_0 ...
            reference:                          varchar(256)  # interf, extref
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

            # get recording folder
            rec_folder = Path(
                (Session & key).fetch1("ephys_ap_data_path")
            ).parent.parent.name

            # get paths and other metadata
            key = _recording.get_recording_filepaths(
                key, rec_metadata, self.recordings_folder, rec_folder
            )
            if key is not None:
                self.insert1(key)

    @schema
    class Unit(dj.Imported):
        precomputed_firing_rate_windows = [33, 100]  # in ms - I think

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
                    triggers = (
                        Session * ValidatedSession * BonsaiTriggers
                        & f'name="{session_name}"'
                    ).fetch1()
                    units = _recording.get_units_firing_rate(
                        units,
                        frate_window,
                        triggers,
                        ValidatedSession.analog_sampling_rate,
                    )
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
            mouse: str, session_name: str, unit_id: int, configuration: str = 'longcol'
        ) -> pd.DataFrame:
            '''
                Gets the recording sites that a unit's spikes are detected on (based on the probe configuration used)
            '''
            rsites = Probe.get_session_sites(mouse, configuration=configuration)
            unit_sites = (
                Unit & f'name="{session_name}"' & f"unit_id={unit_id}"
            ).fetch1("secondary_sites_ids")
            raise NotImplementedError(
                "Double check that this respects probe configurations"
            )

            rsites = rsites.loc[rsites.site_id.isin(unit_sites)]
            return rsites

        def make(self, key):
            recording = (Session * Recording & key).fetch1()

            # load units data
            units = _recording.load_cluster_curation_results(
                recording["spike_sorting_clusters_file_path"],
                recording["spike_sorting_spikes_file_path"],
            )

            # load behavior camera triggers
            triggers = (ValidatedSession * BonsaiTriggers & key).fetch1()

            # deal with concatenated recordinds
            if recording["concatenated"] == 1:
                # load recordings metadata
                rec_metadata = pd.read_excel(
                    Session.recordings_metadata_path, engine="odf"
                )

                # cut unit spikes
                pre_cut, post_cut = _recording.cut_concatenated_units(
                    recording, triggers, rec_metadata
                )
            else:
                pre_cut, post_cut = None, None

            # fill in units
            for nu, unit in enumerate(units):
                logger.debug(f"processing unit {nu+1}/{len(units)}")
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
                    pre_cut=pre_cut,
                    post_cut=post_cut,
                )
                spikes_key = {**key.copy(), **unit_spikes}
                spikes_key["unit_id"] = unit["unit_id"]

                # insert into table
                self.insert1(unit_key)
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
            unit = (Unit * Unit.Spikes & key).fetch1()
            triggers = (ValidatedSession * BonsaiTriggers & key).fetch1()
            logger.info(
                f"Processing: {unit['name']} - unit: {unit['unit_id']}"
            )

            # get firing rates
            for frate_window in Unit.precomputed_firing_rate_windows:
                unit_frate = _recording.get_units_firing_rate(
                    unit,
                    frate_window,
                    triggers,
                    ValidatedSession.analog_sampling_rate,
                )
                frate_key = {**key.copy(), **unit_frate}
                frate_key["unit_id"] = unit["unit_id"]
                frate_key["bin_width"] = frate_window
                del frate_key["site_id"]
                del frate_key["secondary_sites_ids"]
                del frate_key["spikes"]
                del frate_key["spikes_ms"]
                del frate_key["probe_configuration"]

                self.insert1(frate_key)
                # time.sleep(5)

        def check_complete(self):
            # checks that all units have all firing rates
            n_per_unit = len(Unit.precomputed_firing_rate_windows)
            n_units = len(Unit())
            expected = n_per_unit * n_units

            if len(FiringRate()) != expected:
                raise ValueError("Not all units have all firing rates  :(")
            else:
                logger.info("Firing rate has everything")


if __name__ == "__main__":
    # ------------------------------- delete stuff ------------------------------- #
    # ! careful: this is to delete stuff
    # Tracking().drop()
    # LocomotionBouts().drop()
    # Movement().drop()
    # SessionCondition.drop()
    # sys.exit()

    # -------------------------------- sorti filex -----------------------q-------- #

    # logger.info('#####    Sorting FILES')
    # from data.dbase.io import sort_files
    # sort_files()

    # -------------------------------- fill dbase -------------------------------- #

    logger.info("#####    Filling mouse data")
    # Mouse().fill()

    logger.info("#####    Filling Session")
    # Surgery().populate(display_progress=True)
    # Session().fill()
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
    # LocomotionBouts().populate(display_progress=True)
    # Movement().populate(display_progress=True)
    ROICrossing().populate(display_progress=True)
    RoiCrossingsTwins().populate(display_progress=True)

    # ? OPTO
    logger.info("#####    Filling OPTO data")
    # OptoImplant.populate(display_progress=True)6
    # OptoSession.fill()
    # OptoStimuli.populate(display_progress=True)

    # ? EPHYS
    logger.info("#####    Filling Probe")
    # Probe().populate(display_progress=True)
    # Recording().populate(display_progress=True)

    # Unit().populate(display_progress=True)
    # FiringRate().populate(display_progress=True)
    # FiringRate().check_complete()

    # TODO check and debug Opto TABLES
    # TODO make code that takes a locomotion bout that includes a given frame (e.g. to get bouts with opto stim)
    # TODO make clips that show the effects of opto stimulation
    # TODO OptoStimuli should xtract ROI of each stimulus based on tracking data.

    # -------------------------------- print stuff ------------------------------- #
    # print tables contents
    TABLES = [
        Mouse,
        Surgery,
        Session,
        SessionCondition,
        Probe,

    ]
    NAMES = [
        "Mouse",
        "Surgery",
        "Session",
        "SessionCondition",
        "Probe",
    ]
    for tb, name in zip(TABLES, NAMES):
        print_table_content_to_file(tb, name)
