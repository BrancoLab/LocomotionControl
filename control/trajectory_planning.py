import sys

sys.path.append("./")

import numpy as np
from typing import Tuple

from geometry import Path, Vector

from kinematics import track_cordinates_system as TCS


"""
    Extracts a 'center' line representing a track centered through the arena
    and uses an algoirthm from:
        http://phungdinhthang.com/2016/12/16/calculate-racing-lines-automatically/?i=1

    to compute best traces that minimizes length or angle.
"""


# ------------------------------- fit best line ------------------------------ #


class NodeValue:
    def __init__(self, k):
        self.value_coming_from = np.zeros(k)  # value coming from previous
        self.next_best_node = np.zeros(
            k
        )  # index of next best coming from previous


def compute_nodes_values(control_points: dict, k: int, **kwargs) -> dict:
    """
        Given a cost function, it computes the values of each control node based
        on previous/next nodes.

        See: http://phungdinhthang.com/2016/12/16/calculate-racing-lines-automatically/?i=1
    """

    def segment_cost(
        prev: Vector,
        curr: Vector,
        next: Vector,
        angle_cost: float = 0,
        length_cost: float = 1,
    ):
        """
            Cost at one node based on previous and nesxt nodes
        """
        A, B = (curr - prev), (next - curr)
        # compute angle between previous and next segment
        theta = np.radians(A.angle - B.angle)

        # compute arc length
        L = A.magnitude + B.magnitude

        return angle_cost * np.cos(theta) - length_cost * L

    node_values = {
        n: [NodeValue(k) for i in range(k)] for n in control_points.keys()
    }

    for line in range(len(control_points) - 1)[1:][::-1]:
        line_prev, line_curr, line_next = line - 1, line, line + 1

        for c, curr in enumerate(control_points[line_curr]):
            for p, prev in enumerate(control_points[line_prev]):
                best_value = -10000
                best_idx = 0

                for n, nxt in enumerate(control_points[line_next]):
                    # compute the value of curr, coming from prev and going to next
                    value = segment_cost(prev, curr, nxt, **kwargs)

                    # get the value of next coming from curr
                    value += node_values[line_next][n].value_coming_from[c]

                    # keep best
                    if value > best_value:
                        best_value = value
                        best_idx = n

                # update value of coming to current from previous and next best from current coming from previous
                node_values[line_curr][c].value_coming_from[p] = best_value
                node_values[line_curr][c].next_best_node[p] = int(best_idx)

    return node_values


def fit_best_trace(
    control_points: dict,
    center_line: Path,
    k: int,
    angle_cost: float = 1,
    length_cost: float = 1,
) -> Tuple[Path, Path]:
    """
        Fits a best trace through the arena based on the cost for length and angle.
        It returns a Path with the best trace and one with the best trace in the track's coordinates
        system
    """
    node_values = compute_nodes_values(
        control_points, k, angle_cost=angle_cost, length_cost=length_cost
    )

    n = 1
    prev = 0
    node = int(k / 2)
    trace = [node]
    while n < len(node_values) - 1:
        # get the next best node, at this node and coming from the previous
        next_best = node_values[n][node].next_best_node[prev]
        prev = node
        node = int(next_best)
        trace.append(node)

        n += 1

    trace_path = Path(
        [control_points[n][v].x for n, v in enumerate(trace)],
        [control_points[n][v].y for n, v in enumerate(trace)],
    )
    trace_to_track = TCS.path_to_track_coordinates_system(
        center_line, trace_path
    )

    return trace_path, trace_to_track


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from myterial import blue_grey

    import draw
    from kinematics.track import extract_track_from_image

    # ---------------------------------- COMPUTE --------------------------------- #
    K = 11
    points_spacing = 2

    (
        left_line,
        center_line,
        right_line,
        left_to_track,
        center_to_track,
        right_to_track,
        control_points,
    ) = extract_track_from_image(points_spacing=points_spacing, k=K)

    # min_len_trace_path, min_len_trace_to_track = fit_best_trace(
    #     control_points, center_line, K, angle_cost=0, length_cost=1
    # )
    # min_ang_trace_path, min_ang_trace_to_track = fit_best_trace(
    #     control_points, center_line, K, angle_cost=1, length_cost=0
    # )
    # mean_trace_path, mean_trace_to_track = fit_best_trace(
    #     control_points, center_line, K, angle_cost=1, length_cost=.01
    # )

    # ----------------------------------- PLOT ----------------------------------- #
    f = plt.figure(figsize=(8, 12))
    axes = f.subplot_mosaic(
        """
            AAA
            AAA
            AAA
            BBB
        """
    )

    # draw tracking 2d
    draw.Tracking(center_line.x, center_line.y, ax=axes["A"])
    draw.Tracking.scatter(
        center_line.x, center_line.y, ax=axes["A"], color=blue_grey
    )
    draw.Tracking(left_line.x, left_line.y, ls="--", alpha=0.5, ax=axes["A"])
    draw.Tracking(
        right_line.x, right_line.y, ls="--", alpha=0.5, color="b", ax=axes["A"]
    )

    for pts in control_points.values():
        draw.Tracking.scatter(
            pts.x, pts.y, color=[0.7, 0.7, 0.7], ax=axes["A"], s=25, alpha=0.5
        )

    _ = draw.Hairpin(alpha=1, ax=axes["A"])

    # track tracking in track coord system
    draw.Tracking(
        left_to_track.x, left_to_track.y, ax=axes["B"], ls="--", color="k"
    )
    draw.Tracking(
        center_to_track.x, center_to_track.y, ax=axes["B"], lw=2, color="k"
    )
    draw.Tracking(
        right_to_track.x, right_to_track.y, ax=axes["B"], ls="--", color="b"
    )

    # draw best traces
    # draw.Tracking(
    #     min_len_trace_path.x,
    #     min_len_trace_path.y,
    #     color="salmon",
    #     ax=axes["A"],
    #     lw=3,
    # )
    # draw.Tracking(
    #     min_len_trace_to_track.x,
    #     min_len_trace_to_track.y,
    #     color="salmon",
    #     ax=axes["B"],
    # )

    # draw.Tracking(
    #     min_ang_trace_path.x,
    #     min_ang_trace_path.y,
    #     color="green",
    #     ax=axes["A"],
    #     lw=3,
    # )

    # draw.Tracking(
    #     min_ang_trace_to_track.x,
    #     min_ang_trace_to_track.y,
    #     color="green",
    #     ax=axes["B"],
    # )

    # draw.Tracking(
    #     mean_trace_path.x,
    #     mean_trace_path.y,
    #     color="b",
    #     ax=axes["A"],
    #     lw=3,
    # )

    _ = axes["B"].set(ylim=[-5, 5])

    plt.show()
