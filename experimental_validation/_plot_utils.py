from myterial import blue_grey_darker


def draw_tracking(tracking, ax):
    """
        Draws the XY trajectory from tracking

        Arguments:
            tracking: pd.DataFrame with tracking data
    """
    x = tracking.body_x.values
    y = tracking.body_y.values
    ax.plot(x, y, color=blue_grey_darker, lw=0.76, alpha=0.15, zorder=0)
