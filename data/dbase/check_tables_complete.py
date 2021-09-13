from loguru import logger
import sys

from tpd import recorder

sys.path.append("./")
from data import dbase


def get_sessions_to_track(recordings_only: bool = True):
    """
        It gets a list of sessions left to track
    """
    logger.info("Getting sessions that need to be tracked with DLC still")
    sessions_in_table = dbase.db_tables.Session.fetch("name")

    if recordings_only:
        sessions_in_table = [
            s
            for s in sessions_in_table
            if dbase.db_tables.Session.has_recording(s)
        ]

    need_tracking = []
    for session in sessions_in_table:
        if not dbase.db_tables.Session.was_tracked(session):
            need_tracking.append(session)

    logger.info(
        f"Found {len(need_tracking)}/{len(sessions_in_table)} sessions that still need to be tracked"
    )
    recorder.add_text(", ".join(need_tracking), name="__to_track")


if __name__ == "__main__":
    get_sessions_to_track()
