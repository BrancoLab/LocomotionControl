import sys
from tpd import recorder
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("./")
from pathlib import Path

from fcutils.plot.figure import clean_axes
from fcutils.maths import derivative

from analysis.visuals import plot_balls_errors, plot_tracking_xy

from data.dbase import db_tables
from data.dbase.hairpin_trace import HairpinTrace
from data import data_utils

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

"""
    Plot a few exploratory plots for each session a recording was performed on.

        1. Overview of tracking data  
"""


hairpin = HairpinTrace()

sessions = (
    db_tables.ValidatedSession * db_tables.Session
    & "is_recording=1"
    & 'arena="hairpin"'
)

for session in sessions:
    # start a new recorder sesssion
    recorder.start(
        base_folder=base_folder, name=session["name"], timestamp=False
    )

    # load tracking data
    tracking = db_tables.Tracking.get_session_tracking(
        session["name"], body_only=False
    )
    body_tracking = tracking.loc[tracking.bpname == "body"].iloc[0]
    data_utils.downsample_tracking_data(body_tracking, factor=10)

    # bin tracking data by arena segment
    binned_tracking = data_utils.bin_tracking_data_by_arena_position(
        body_tracking
    )

    # ! get when moving towards the goal
    when_heading_to_goal = np.where(
        (derivative(body_tracking["global_coord"]) > 0)
        & (body_tracking["speed"] > 0.05)
    )[0]
    heading_to_goal = data_utils.select_by_indices(
        body_tracking, when_heading_to_goal
    )
    binned_heading_to_goal = data_utils.bin_tracking_data_by_arena_position(
        heading_to_goal
    )

    # ! get when moving away the goal
    when_heading_away_goal = np.where(
        (derivative(body_tracking["global_coord"]) < 0)
        & (body_tracking["speed"] > 0.05)
    )[0]
    heading_away_goal = data_utils.select_by_indices(
        body_tracking, when_heading_away_goal
    )
    binned_heading_away_goal = data_utils.bin_tracking_data_by_arena_position(
        heading_away_goal
    )

    # plt tracking data
    f = plt.figure(figsize=(19, 18))
    axes_dict = f.subplot_mosaic(
        """
        ABC
        ABE
        FDG
        FDH

    """
    )
    f.suptitle(session["name"])
    f._save_name = "tracking_data_2d"

    # draw tracking colored by arena segment
    hairpin.draw(axes_dict["A"], body_tracking)

    # draw tracking colored by orientation
    plot_tracking_xy(
        heading_to_goal,
        "orientation",
        ax=axes_dict["F"],
        vmin=0,
        vmax=360,
        cmap="bwr",
    )
    plot_tracking_xy(
        heading_away_goal,
        "orientation",
        ax=axes_dict["D"],
        vmin=0,
        vmax=360,
        cmap="bwr",
    )

    # draw linearized trackign in time
    c = hairpin.colors_from_segment(body_tracking["segment"])
    y = np.linspace(1, 0, len(c))
    axes_dict["B"].scatter(body_tracking["global_coord"], y, c=c)

    # plot speed by position along arena
    plot_balls_errors(
        hairpin.X,
        binned_heading_to_goal["speed"],
        binned_heading_to_goal["speed_std"],
        hairpin.colors,
        axes_dict["C"],
    )
    axes_dict["C"].axhline(
        np.nanmean(binned_heading_to_goal["speed"]), lw=2, ls=":", zorder=-1
    )

    plot_balls_errors(
        hairpin.X,
        binned_heading_away_goal["speed"],
        binned_heading_away_goal["speed_std"],
        hairpin.colors,
        axes_dict["E"],
    )
    axes_dict["E"].axhline(
        np.nanmean(binned_heading_away_goal["speed"]), lw=2, ls=":", zorder=-1
    )

    # plot orientation by position when going towards goal
    plot_balls_errors(
        hairpin.X,
        binned_heading_to_goal["orientation"],
        binned_heading_to_goal["orientation_std"],
        hairpin.colors,
        axes_dict["G"],
    )

    # plot orientation by position when going away goal
    plot_balls_errors(
        hairpin.X,
        binned_heading_away_goal["orientation"],
        binned_heading_away_goal["orientation_std"],
        hairpin.colors,
        axes_dict["H"],
    )

    # cleanup and save
    clean_axes(f)

    axes_dict["A"].set(xlabel="xpos (cm)", ylabel="ypos (cm)")
    axes_dict["B"].set(ylabel="time in exp", xlabel="arena position")
    axes_dict["C"].set(
        xlabel="arena position", ylabel="running speed to goal cm/s"
    )
    axes_dict["D"].set(ylabel="orient mving away goal")
    axes_dict["E"].set(
        xlabel="arena position", ylabel="running speed away goal cm/s"
    )
    axes_dict["F"].set(ylabel="orient mving twrds goal")
    axes_dict["G"].set(ylabel="avg orientation towards goal")
    axes_dict["H"].set(ylabel="avg orientation away goal")

    for ax in "ABFD":
        axes_dict[ax].axis("equal")
    axes_dict["H"].axis("off")

    plt.show()
    # break

    recorder.add_figures(svg=False)
    plt.close("all")
