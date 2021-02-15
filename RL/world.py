import numpy as np

from fcutils.maths.coordinates import cart2pol
from fcutils.maths.geometry import calc_distance_between_points_2d as dist

# from control.config import dt
from control import trajectories
from control.model import state


class RLWorld:
    route_progression = 0
    curr_traj_waypoint_idx = 0
    n_waypoints_ahead = 0

    def __init__(self):
        self.trajectory = trajectories.simulated()[1]

        self.curr_traj_waypoint_idx = 1  #  keep track of traj progression

    def reset(self):
        self.trajectory = trajectories.simulated()[1]
        self.curr_traj_waypoint_idx = 1
        self.route_progression = 0

    def get_delta_position(self, agent):
        """
            Get the dx dy of the agents position
            vs the current state in the trajectory
        """
        # get traj idx
        curr_idx = self.current_trajectory_idx(agent)

        # get trajectory at this point
        traj = self.trajectory[curr_idx]

        # convert to polar coordinates
        dx = traj[0] - agent.curr_x.x
        dy = traj[1] - agent.curr_x.y

        return dx, dy

    def current_trajectory_idx(self, agent):
        """
            Gets the index of the trajectory waypoint the
            agent is closest to.
        """

        # get the closest traj point in the next chunk of trajectory
        curr_idx = self.curr_traj_waypoint_idx
        sel_traj = self.trajectory[curr_idx - 1 : curr_idx + 30, :2]
        dst = np.linalg.norm(agent.curr_x[:2] - sel_traj, axis=1)
        min_idx = np.argmin(dst) + curr_idx
        self.curr_traj_waypoint_idx = min_idx
        return min_idx + self.n_waypoints_ahead

    def get_next_waypint(self):
        return state(*self.trajectory[self.curr_traj_waypoint_idx])

    def get_agent_trajectory_input(self, agent):
        """
            Based on where the agent is along the trajectory, 
            it returns the distance and angle to the next waypoint
        """
        # get dx dy
        dx, dy = self.get_delta_position(agent)

        # convert to polar coordinates
        r, psy = cart2pol(dx, dy)

        # get speeds deltas
        curr_idx = self.current_trajectory_idx(agent)

        # get trajectory at this point
        traj = self.trajectory[curr_idx]
        dv = traj[3] - agent.curr_x.v
        domega = traj[4] - agent.curr_x.omega

        return r, psy, dv, domega

    def get_reward(self, agent, state_idx, controls):
        """
            Get the reward as the inverse of the state error
        """
        # get traj idx
        dx, dy = self.get_delta_position(agent)

        progression = self.curr_traj_waypoint_idx / len(self.trajectory)
        progression_improvement = progression - self.route_progression
        self.route_progression = progression

        # cost = np.linalg.norm(controls) * dt * 0.2

        reward = progression_improvement  # - cost

        return dx, dy, reward

    def isdone(self, agent, min_dist=10):
        # check if we are too far off the trajectory
        dx, dy = self.get_delta_position(agent)
        if dx > 5 or dy > 5:
            self.progress.log(
                "WORLD", "Trial done because too far from the trajectory"
            )
            return True

        # get distance to trajectory end point
        traj = self.trajectory[-1]

        dx = traj[0] - agent.curr_x.x
        dy = traj[1] - agent.curr_x.y

        # check if we are at the end of the trajectory
        dst = dist(dx, dy)
        if dst <= min_dist:
            self.progress.log(
                "WORLD", "Trial done because at end of trajectory"
            )
            return True
        else:
            return False
