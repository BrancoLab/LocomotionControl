import sys
import matplotlib.patheffects as path_effects
from matplotlib.artist import Artist
import pandas as pd
from loguru import logger


sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

from data.dbase.db_tables import (
    Probe,
    Unit,
    Recording,
    Tracking,
    LocomotionBouts,
    SessionCondition,
)


def get_recording_names():
    """
        Get the names of the recordings (M2)
    """
    return (Recording * Probe & "target='MOs'").fetch("name")


def get_data(recording: str):
    """
        Get all relevant data for a recording.
        Gets the ephys data and tracking data for all limbs
    """
    tracking = Tracking.get_session_tracking(
        recording, body_only=False, movement=True
    )
    left_fl = tracking.loc[tracking.bpname == "left_fl"].iloc[0]
    right_fl = tracking.loc[tracking.bpname == "right_fl"].iloc[0]
    left_hl = tracking.loc[tracking.bpname == "left_hl"].iloc[0]
    right_hl = tracking.loc[tracking.bpname == "right_hl"].iloc[0]
    body = tracking.loc[tracking.bpname == "body"].iloc[0]

    logger.info(f"Got tracking data for {recording}")

    # get units
    recording = (Recording & f"name='{recording}'").fetch1()
    cf = recording["recording_probe_configuration"]
    units = Unit.get_session_units(
        recording["name"],
        cf,
        spikes=True,
        firing_rate=False,
        frate_window=100,
    )
    if len(units):
        units = units.sort_values("brain_region", inplace=False).reset_index()
        logger.info(f"Got {len(units)} units for {recording['name']}")
    else:
        logger.info(f"No units for {recording['name']}")

    return units, left_fl, right_fl, left_hl, right_hl, body


def get_session_bouts(
    session: str, complete: str = "true", direction: str = "outbound"
):
    """
        Get bouts (complete/all - in/out bound) for a session
    """

    bouts = pd.DataFrame(
        (
            LocomotionBouts * SessionCondition
            & f'name="{session}"'
            & f'complete="{complete}"'
            & f'direction="{direction}"'
        ).fetch()
    )
    logger.info(
        f"Got {len(bouts)} bouts for {session} | {complete} | {direction}"
    )
    return bouts


# ---------------------------------------------------------------------------- #
#                                    VISUALS                                   #
# ---------------------------------------------------------------------------- #


def outline(artist: Artist, lw: float = 1, color: str = "white"):
    artist.set_path_effects(
        [path_effects.withStroke(linewidth=lw, foreground=color,)]
    )
