from math import sin, cos, atan2, sqrt, pi, hypot, acos
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import sys

sys.path.append("./")


from control.paths.utils import mod2pi, pi_2_pi
from control.paths.waypoints import Waypoints, Waypoint
from geometry import Path

"""
    Code adapted from: https://github.com/zhm-real/CurvesGenerator
    zhm-real shared code to create different types of paths under an MIT license.
    The logic of the code is left un-affected here, I've just refactored it.
"""


# utility
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


# ---------------------------------------------------------------------------- #
#                                   PLANNERS                                   #
# ---------------------------------------------------------------------------- #
class Planner:
    tag = ["N", "a", "N"]

    def __call__(self, alpha, beta, dist):
        t, p, q = self.fit(alpha, beta, dist)
        return t, p, q, self.tag

    @staticmethod
    def _calc_sines(alpha, beta):
        return (
            sin(alpha),
            sin(beta),
            cos(alpha),
            cos(beta),
            cos(alpha - beta),
        )

    def fit(self, alpha, beta, dist):
        raise NotImplementedError


class LSL(Planner):
    tag = ["L", "S", "L"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)

        if p_lsl < 0:
            return (
                None,
                None,
                None,
            )
        else:
            p_lsl = sqrt(p_lsl)

        denominate = dist + sin_a - sin_b
        t_lsl = mod2pi(-alpha + atan2(cos_b - cos_a, denominate))
        q_lsl = mod2pi(beta - atan2(cos_b - cos_a, denominate))

        return (
            t_lsl,
            p_lsl,
            q_lsl,
        )


class RSR(Planner):
    tag = ["R", "S", "R"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)

        if p_rsr < 0:
            return None, None, None
        else:
            p_rsr = sqrt(p_rsr)

        denominate = dist - sin_a + sin_b
        t_rsr = mod2pi(alpha - atan2(cos_a - cos_b, denominate))
        q_rsr = mod2pi(-beta + atan2(cos_a - cos_b, denominate))

        return t_rsr, p_rsr, q_rsr


class LSR(Planner):
    tag = ["L", "S", "R"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)

        if p_lsr < 0:
            return None, None, None
        else:
            p_lsr = sqrt(p_lsr)

        rec = atan2(-cos_a - cos_b, dist + sin_a + sin_b) - atan2(-2.0, p_lsr)
        t_lsr = mod2pi(-alpha + rec)
        q_lsr = mod2pi(-mod2pi(beta) + rec)

        return t_lsr, p_lsr, q_lsr


class RSL(Planner):
    tag = ["R", "S", "L"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)

        if p_rsl < 0:
            return None, None, None
        else:
            p_rsl = sqrt(p_rsl)

        rec = atan2(cos_a + cos_b, dist - sin_a - sin_b) - atan2(2.0, p_rsl)
        t_rsl = mod2pi(alpha - rec)
        q_rsl = mod2pi(beta - rec)

        return t_rsl, p_rsl, q_rsl


class RLR(Planner):
    tag = ["R", "L", "R"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        rec = (
            6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)
        ) / 8.0

        if abs(rec) > 1.0:
            return None, None, None

        p_rlr = mod2pi(2 * pi - acos(rec))
        t_rlr = mod2pi(
            alpha
            - atan2(cos_a - cos_b, dist - sin_a + sin_b)
            + mod2pi(p_rlr / 2.0)
        )
        q_rlr = mod2pi(alpha - beta - t_rlr + mod2pi(p_rlr))

        return t_rlr, p_rlr, q_rlr


class LRL(Planner):
    tag = ["L", "R", "L"]

    def __init__(self):
        super().__init__()

    def fit(self, alpha, beta, dist):
        sin_a, sin_b, cos_a, cos_b, cos_a_b = self._calc_sines(alpha, beta)

        rec = (
            6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_b - sin_a)
        ) / 8.0

        if abs(rec) > 1.0:
            return None, None, None

        p_lrl = mod2pi(2 * pi - acos(rec))
        t_lrl = mod2pi(
            -alpha - atan2(cos_a - cos_b, dist + sin_a - sin_b) + p_lrl / 2.0
        )
        q_lrl = mod2pi(mod2pi(beta) - alpha - t_lrl + mod2pi(p_lrl))

        return t_lrl, p_lrl, q_lrl


# ---------------------------------------------------------------------------- #
#                                   DUBIN                                      #
# ---------------------------------------------------------------------------- #


class DubinPath:
    def __init__(self, waypoints: Waypoints, max_curvature: float = 0.2):
        self.waypoints = waypoints
        self.max_curvature = max_curvature

        self.planners = [LSL(), RSR(), LSR(), RSL(), RLR(), LRL()]

        # store path variables
        self.x = []
        self.y = []
        self.theta = []
        self.lengths = []  # length of each segment
        self.mode = []  # type of each segment

    def generate_local_course(self, L, lengths, mode, step_size: float = 0.1):
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
                    ind,
                    pd,
                    m,
                    self.max_curvature,
                    ox,
                    oy,
                    oyaw,
                    px,
                    py,
                    pyaw,
                    directions,
                )
                pd += d

            ll = l - pd - d  # calc remain length

            ind += 1
            px, py, pyaw, directions = interpolate(
                ind,
                l,
                m,
                self.max_curvature,
                ox,
                oy,
                oyaw,
                px,
                py,
                pyaw,
                directions,
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

    def planning_from_origin(self, gx, gy, gtheta):
        D = hypot(gx, gy)
        d = D * self.max_curvature

        theta = mod2pi(atan2(gy, gx))
        alpha = mod2pi(-theta)
        beta = mod2pi(gtheta - theta)

        best_cost = float("inf")
        bt, bp, bq, best_mode = None, None, None, None

        for planner in self.planners:
            t, p, q, mode = planner(alpha, beta, d)

            if t is None:
                continue

            cost = abs(t) + abs(p) + abs(q)
            if best_cost > cost:
                bt, bp, bq, best_mode = t, p, q, mode
                best_cost = cost
        lengths = [bt, bp, bq]

        x_list, y_list, theta_list, directions = self.generate_local_course(
            sum(lengths), lengths, best_mode,
        )

        return x_list, y_list, theta_list, best_mode, best_cost

    def fit_segment(self, wp1: Waypoint, wp2: Waypoint):
        gx = wp2.x - wp1.x
        gy = wp2.y - wp1.y

        theta1, theta2 = np.radians(wp1.theta), np.radians(wp2.theta)

        l_rot = Rot.from_euler("z", theta1).as_matrix()[0:2, 0:2]
        le_xy = np.stack([gx, gy]).T @ l_rot
        le_theta = theta2 - theta1

        lp_x, lp_y, lp_theta, mode, lengths = self.planning_from_origin(
            le_xy[0], le_xy[1], le_theta
        )

        rot = Rot.from_euler("z", -theta1).as_matrix()[0:2, 0:2]
        converted_xy = np.stack([lp_x, lp_y]).T @ rot
        x = converted_xy[:, 0] + wp1.x
        y = converted_xy[:, 1] + wp1.y
        theta = [pi_2_pi(i_yaw + theta1) for i_yaw in lp_theta]

        # add to path
        self.x.extend(list(x)[1:-1])
        self.y.extend(list(y)[1:-1])
        self.theta.extend(list(theta)[1:-1])
        self.mode.extend(list(mode)[1:-1])
        self.lengths.append(lengths)

    def fit(self) -> Path:
        for i in range(len(self.waypoints) - 1):
            self.fit_segment(self.waypoints[i], self.waypoints[i + 1])
        return Path(self.x, self.y, self.theta)


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    import matplotlib.pyplot as plt
    import pandas as pd

    import draw

    f, ax = plt.subplots(figsize=(7, 10))

    # load and draw tracking data
    from fcutils.path import files

    for fp in files(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/control/",
        "*.h5",
    ):
        tracking = pd.read_hdf(fp, key="hdf")
        tracking.x = 20 - tracking.x + 20
        # draw.Tracking.scatter(tracking.x, tracking.y, c=tracking.theta, vmin=0, vmax=360, cmap='bwr', lw=1, ec='k')
        draw.Tracking(tracking.x, tracking.y, alpha=0.7)

    # draw hairpin arena
    draw.Hairpin(ax)

    # draw waypoints
    wps = Waypoints()
    for wp in wps:
        draw.Arrow(wp.x, wp.y, wp.theta, 2, width=4, color="g")

    # fit dubin path
    dubin = DubinPath(wps).fit()
    draw.Tracking(dubin.x, dubin.y, lw=2, color="k")
    plt.show()
