from dataclasses import dataclass
from myterial import salmon, blue_light, red_light, green, indigo, teal
from typing import Optional


"""
    Define the position of ROIs in the arena
"""


@dataclass
class ROI:
    name: str
    x_0: int  # position of top left corner in cm
    y_0: int
    x_1: int  # position of top right corner in cm
    y_1: int
    g_0: float  # start in global coord
    g_1: float  # end in global coord
    color: str
    turn_g_start: Optional[float] = None  # G of just before start (> g_0)
    turn_g_apex: Optional[float] = None  # G of apex of turn (> g_turn)


T1 = ROI(
    "T1",
    x_0=8,
    x_1=24,
    y_0=0,
    y_1=20,
    g_0=0.02,
    g_1=0.27,
    color=salmon,
    turn_g_start=0.03,
    turn_g_apex=0.14,
)
T2 = ROI(
    "T2",
    x_0=8,
    x_1=32,
    y_0=33,
    y_1=52,
    g_0=0.20,
    g_1=0.48,
    color=blue_light,
    turn_g_start=0.22,
    turn_g_apex=0.34,
)
T3 = ROI(
    "T3",
    x_0=24,
    x_1=40,
    y_0=0,
    y_1=20,
    g_0=0.416,
    g_1=0.66,
    color=red_light,
    turn_g_start=0.43,
    turn_g_apex=0.52,
)
T4 = ROI(
    "T4",
    x_0=24,
    x_1=40,
    y_0=0,
    y_1=20,
    g_0=0.6,
    g_1=945,
    color=indigo,
    turn_g_start=0.625,
    turn_g_apex=0.73,
)

S1 = ROI("S1", x_0=32, x_1=40, y_0=20, y_1=45, g_0=0.585, g_1=0.7, color=green)
S2 = ROI("S2", x_0=32, x_1=40, y_0=20, y_1=45, g_0=0.855, g_1=0.99, color=teal)

ROIs = [T1, T2, T3, T4, S1, S2]
ROIs_dict = dict(T1=T1, T2=T2, T3=T3, T4=T4, S1=S1, S2=S2,)
