from myterial import blue_grey_light

from draw.tracking import Tracking
from geometry import Path


def draw_track(center: Path, left: Path, right: Path, **kwargs):
    """
        Draws a center line and two side paths.
        This works for e.g. linearized track + extruded sides
    """
    Tracking(center.x, center.y, lw=2, color=blue_grey_light, **kwargs)
    Tracking(left.x, left.y, lw=1, ls="--", color=blue_grey_light, **kwargs)
    Tracking(right.x, right.y, lw=1, ls="--", color=blue_grey_light, **kwargs)
