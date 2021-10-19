from math import sin, cos, atan2, sqrt, pi, hypot, acos
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from loguru import logger

import sys

sys.path.append("./")


from control.utils import mod2pi, pi_2_pi

# ---------------------------------------------------------------------------- #
#                                   WAYPOINTS                                  #
# ---------------------------------------------------------------------------- #
@dataclass
class Waypoint:
    x: int
    y: int
    theta: int


class Waypoints:
    waypoints = [
        Waypoint(20, 40, 270),  # start
        Waypoint(20, 10, 270),  # 1st bend
        Waypoint(12, 8, 110),  # 1st bend
        Waypoint(12, 35, 90),  # 2nd bend
        Waypoint(20, 45, 0),  # 2nd bend - halfway
        Waypoint(28, 35, 270),  # 2nd bend
        Waypoint(28, 12, 270),  # 3nd bend
        Waypoint(36, 10, 90),  # 3nd bend
        Waypoint(36, 46, 90),  # 4nd bend
        Waypoint(28, 55, 180),  # 4nd bend - halfway
        Waypoint(15, 55, 180),  # 4nd bend - halfway
        Waypoint(6, 46, 270),  # 4nd bend
        Waypoint(6, 4, 270),  # end
    ]
    _idx = 0

    @property
    def x(self) -> np.ndarray:
        return np.array([wp.x for wp in self.waypoints])

    @property
    def y(self) -> np.ndarray:
        return np.array([wp.y for wp in self.waypoints])

    @property
    def theta(self) -> np.ndarray:
        return np.array([wp.theta for wp in self.waypoints])

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.waypoints):
            self._idx += 1
            return self.waypoints[self._idx - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, item):
        return self.waypoints[item]


# ---------------------------------------------------------------------------- #
#                                   DUBIN                                      #
# ---------------------------------------------------------------------------- #


def LSL(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)

    if p_lsl < 0:
        return None, None, None, ["L", "S", "L"]
    else:
        p_lsl = sqrt(p_lsl)

    denominate = dist + sin_a - sin_b
    t_lsl = mod2pi(-alpha + atan2(cos_b - cos_a, denominate))
    q_lsl = mod2pi(beta - atan2(cos_b - cos_a, denominate))

    return t_lsl, p_lsl, q_lsl, ["L", "S", "L"]


def RSR(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)

    if p_rsr < 0:
        return None, None, None, ["R", "S", "R"]
    else:
        p_rsr = sqrt(p_rsr)

    denominate = dist - sin_a + sin_b
    t_rsr = mod2pi(alpha - atan2(cos_a - cos_b, denominate))
    q_rsr = mod2pi(-beta + atan2(cos_a - cos_b, denominate))

    return t_rsr, p_rsr, q_rsr, ["R", "S", "R"]


def LSR(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)

    if p_lsr < 0:
        return None, None, None, ["L", "S", "R"]
    else:
        p_lsr = sqrt(p_lsr)

    rec = atan2(-cos_a - cos_b, dist + sin_a + sin_b) - atan2(-2.0, p_lsr)
    t_lsr = mod2pi(-alpha + rec)
    q_lsr = mod2pi(-mod2pi(beta) + rec)

    return t_lsr, p_lsr, q_lsr, ["L", "S", "R"]


def RSL(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)

    if p_rsl < 0:
        return None, None, None, ["R", "S", "L"]
    else:
        p_rsl = sqrt(p_rsl)

    rec = atan2(cos_a + cos_b, dist - sin_a - sin_b) - atan2(2.0, p_rsl)
    t_rsl = mod2pi(alpha - rec)
    q_rsl = mod2pi(beta - rec)

    return t_rsl, p_rsl, q_rsl, ["R", "S", "L"]


def RLR(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    rec = (
        6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)
    ) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["R", "L", "R"]

    p_rlr = mod2pi(2 * pi - acos(rec))
    t_rlr = mod2pi(
        alpha
        - atan2(cos_a - cos_b, dist - sin_a + sin_b)
        + mod2pi(p_rlr / 2.0)
    )
    q_rlr = mod2pi(alpha - beta - t_rlr + mod2pi(p_rlr))

    return t_rlr, p_rlr, q_rlr, ["R", "L", "R"]


def LRL(alpha, beta, dist):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_a_b = cos(alpha - beta)

    rec = (
        6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_b - sin_a)
    ) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["L", "R", "L"]

    p_lrl = mod2pi(2 * pi - acos(rec))
    t_lrl = mod2pi(
        -alpha - atan2(cos_a - cos_b, dist + sin_a - sin_b) + p_lrl / 2.0
    )
    q_lrl = mod2pi(mod2pi(beta) - alpha - t_lrl + mod2pi(p_lrl))

    return t_lrl, p_lrl, q_lrl, ["L", "R", "L"]


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * cos(oyaw)
        py[ind] = oy + l / maxc * sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = sin(l) / maxc
        if m == "L":
            ldy = (1.0 - cos(l)) / maxc
        elif m == "R":
            ldy = (1.0 - cos(l)) / (-maxc)

        gdx = cos(-oyaw) * ldx + sin(-oyaw) * ldy
        gdy = -sin(-oyaw) * ldx + cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(
                ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
            )
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = interpolate(
            ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
        )

    if len(px) <= 1:
        return [], [], [], []

    # remove unused data
    while len(px) >= 1 and px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def planning_from_origin(gx, gy, gyaw, curv, step_size):
    D = hypot(gx, gy)
    d = D * curv

    theta = mod2pi(atan2(gy, gx))
    alpha = mod2pi(-theta)
    beta = mod2pi(gyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)

        if t is None:
            continue

        cost = abs(t) + abs(p) + abs(q)
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]

    x_list, y_list, yaw_list, directions = generate_local_course(
        sum(lengths), lengths, best_mode, curv, step_size
    )

    return x_list, y_list, yaw_list, best_mode, best_cost


def calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, curv, step_size=0.1):
    gx = gx - sx
    gy = gy - sy

    l_rot = Rot.from_euler("z", syaw).as_dcm()[0:2, 0:2]
    le_xy = np.stack([gx, gy]).T @ l_rot
    le_yaw = gyaw - syaw

    lp_x, lp_y, lp_yaw, mode, lengths = planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, curv, step_size
    )

    rot = Rot.from_euler("z", -syaw).as_dcm()[0:2, 0:2]
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + sx
    y_list = converted_xy[:, 1] + sy
    yaw_list = [pi_2_pi(i_yaw + syaw) for i_yaw in lp_yaw]

    return PATH(lengths, mode, x_list, y_list, yaw_list)


class PATH:
    def __init__(self, L, mode, x, y, yaw):
        self.L = L  # total path length [float]
        self.mode = mode  # type of each part of the path [string]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    import matplotlib.pyplot as plt

    import draw

    f, ax = plt.subplots(figsize=(7, 10))

    # draw hairpin arena
    draw.Hairpin(ax)

    # draw waypoints
    wps = Waypoints()
    for wp in wps:
        draw.Arrow(ax, wp.x, wp.y, wp.theta, 2, width=4, color="r")

    # iterate over each segment and get dubin element
    max_c = 0.25  # max curvature
    path_x, path_y, yaw = [], [], []
    for i in range(len(wps) - 1):
        logger.info(f"Processing item {i}")
        path_i = calc_dubins_path(
            wps[i].x,
            wps[i].y,
            np.radians(wps[i].theta),
            wps[i + 1].x,
            wps[i + 1].y,
            np.radians(wps[i + 1].theta),
            max_c,
        )

        for x, y, iyaw in zip(path_i.x, path_i.y, path_i.yaw):
            path_x.append(x)
            path_y.append(y)
            yaw.append(iyaw)

    draw.Tracking(ax, path_x, path_y, lw=2, color="k")
    plt.show()
