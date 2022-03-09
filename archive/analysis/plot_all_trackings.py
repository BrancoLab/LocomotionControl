import sys

sys.path.append("./")

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pathlib import Path

from tpd import recorder
from myterial import salmon, teal, indigo

import draw
from data.dbase.db_tables import Tracking, ValidatedSession
from fcutils.progress import track


folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")
recorder.start(base_folder=folder, folder_name="all_tracking", timestamp=False)


"""
    Plot the tracking for each session to check that everything is ok
"""

sessions = ValidatedSession().fetch("name")

for session in track(sessions):
    save_name = folder / "all_tracking" / (session + ".png")
    if save_name.exists() or "open" in session:
        continue

    tracking = Tracking.get_session_tracking(
        session, body_only=False, movement=False
    )

    if tracking.empty:
        logger.info(f'"{session}" - no tracking')
        continue

    body = tracking.loc[tracking.bpname == "body"].iloc[0]
    snout = tracking.loc[tracking.bpname == "snout"].iloc[0]
    paw = tracking.loc[tracking.bpname == "right_fl"].iloc[0]
    tail = tracking.loc[tracking.bpname == "tail_base"].iloc[0]

    f, ax = plt.subplots(figsize=(9, 12))

    draw.Hairpin(ax=ax)
    draw.Tracking(body.x, body.y, ax=ax)

    for bp, color in zip((snout, paw, tail), (teal, salmon, indigo)):
        draw.Tracking.scatter(
            bp.x[::30],
            bp.y[::30],
            color=color,
            alpha=0.5,
            zorder=100,
            ax=ax,
            label=bp.bpname,
        )

    ax.set(title=f"{session} - {len(tracking.x)} frames")
    ax.legend()

    recorder.add_figure(f, session, svg=False)

    plt.close(f)
