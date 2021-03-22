import datajoint as dj
from pathlib import Path
import shutil
from loguru import logger
import numpy as np

from fcutils.path import from_yaml, files
from fcutils.progress import track
from fcutils.video import get_video_params
from fcutils.maths.signals import get_onset_offset

import sys

sys.path.append("./")
from experimental_validation.hairpin import schema


datafolder = Path("Z:\\swc\\branco\\Federico\\Locomotion\\raw")


# ----------------------------------- utils ---------------------------------- #


def sort_files():
    """ sorts raw files into the correct folders """
    logger.info("Sorting raw files")
    fls = files(datafolder / "tosort")

    raise NotImplementedError("Bonsai should save .csv files too now")

    if isinstance(fls, list):
        for f in fls:
            src = datafolder / "tosort" / f.name

            if f.suffix == ".avi":
                dst = datafolder / "video" / f.name
            elif f.suffix == ".bin" or f.suffix == ".csv":
                dst = datafolder / "analog_inputs" / f.name
            else:
                logger.info(f"File not recognized: {f}")
                continue

            logger.info(f"Moving file {src} to {dst}")
        shutil.move(src, dst)


# For manual tables
def insert_entry_in_table(dataname, checktag, data, table, overwrite=False):
    """
        Tries to add an entry to a databse table taking into account entries already in the table

        dataname: value of indentifying key for entry in table
        checktag: name of the identifying key ['those before the --- in the table declaration']
        data: entry to be inserted into the table
        table: database table
    """
    if dataname in list(table.fetch(checktag)):
        return

    try:
        table.insert1(data)
        logger.debug("     ... inserted {} in table".format(dataname))
    except:
        if dataname in list(table.fetch(checktag)):
            logger.debug("Entry with id: {} already in table".format(dataname))
        else:
            logger.debug(table)
            raise ValueError(
                "Failed to add data entry {}-{} to {} table".format(
                    checktag, dataname, table.full_table_name
                )
            )


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
    """

    def fill(self):
        data = from_yaml("experimental_validation\hairpin\dbase\sessions.yaml")
        logger.info("Filling in session table")

        for session in track(
            data, description="Adding sessions", transient=True
        ):
            key = dict(mouse_id=session["mouse"], name=session["name"])

            # get file paths
            key["video_file_path"] = (
                datafolder / "video" / (session["name"] + "_video.avi")
            )
            key["ai_file_path"] = (
                datafolder
                / "analog_inputs"
                / (session["name"] + "_analog.bin")
            )

            key["csv_file_path"] = (
                datafolder / "analog_inputs" / (session["name"] + "_data.csv")
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


# ---------------------------------------------------------------------------- #
#                              validated sessions                              #
# ---------------------------------------------------------------------------- #


@schema
class ValidatedSessions(dj.Imported):
    definition = """
        # checks that the video and AI files for a session are saved correctly
        -> Session
    """
    analog_sampling_rate = 30000

    def make(self, key):
        session = (Session & key).fetch1()

        # load video and get metadata
        nframes, w, h, fps, _ = get_video_params(session["video_file_path"])
        if fps != 60:
            raise ValueError("Expected video FPS: 60")

        # load analog
        analog = np.fromfile(session["ai_file_path"], dtype=np.double).reshape(
            -1, 4
        )

        # check that the number of frames is correct
        frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
        if len(frame_trigger_times) != nframes:
            raise ValueError(
                f'session: {session["name"]} - found {nframes} video frames and {frame_trigger_times} trigger times in analog input'
            )

        # check that the number of frames is what you'd expect given the duration of the exp
        first_frame_s = frame_trigger_times[0] / self.analog_sampling_rate
        last_frame_s = frame_trigger_times[-1] / self.analog_sampling_rate
        exp_dur = last_frame_s - first_frame_s  # video duration in seconds
        expected_n_frames = np.floor(exp_dur * 60).astype(np.int64)
        if np.abs(expected_n_frames - nframes) > 5:
            raise ValueError(
                f"[b yellow]Expected {expected_n_frames} frames but found {nframes} in video"
            )

        # all OK, add to table to avoid running again in the future
        self.insert1(key)


@schema
class SessionData(dj.Imported):
    definition = """
        # stores AI and csv data in a nicely formatted manner
        -> Session:
        lick_sensor: longblob
        speaker_signal: longblob
        pump: longblob
        roi_activity: longblob
        mouse_in_roi: longblob
        reward_signal: longblob
    """
    analog_sampling_rate = 30000

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
