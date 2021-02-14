import numpy as np
import matplotlib.pyplot as plt

from rich.live import Live

from myterial import orange, salmon
from pyrnn._utils import GracefulInterruptHandler

from RL.agent import RLAgent
from RL.world import RLWorld
from RL import settings
from RL.progress import Progress, Variables, State


class Manager:
    def __init__(self):
        self.agent = RLAgent()
        self.world = RLWorld()

    def get_state(self):
        """
            Gets the input to the agent at a given
            step in a simulation
        """
        # get the current trajectory input
        r, psy = self.world.get_agent_trajectory_input(self.agent)

        # get the agent's state
        v, omega = self.agent.curr_x.v, self.agent.curr_x.omega

        return np.array([r, psy, v, omega])

    def train(self):
        variables = Variables()
        state_live = State()
        progress = Progress(self.agent, variables, state_live)
        progress.log("START", "Starting trainig")
        tid = progress.add_task(
            description=f"[{salmon}]Training", total=settings.N_EPISODES
        )

        self.world.progress = progress

        scores = []
        with Live(progress.layout, screen=True, refresh_per_second=60) as live:
            with GracefulInterruptHandler() as h:
                for n in range(settings.N_EPISODES):
                    t2id = progress.add_task(
                        description=f"[{orange}]Trial {n+1}",
                        total=settings.MAX_EPISODE_LEN,
                    )

                    self.world.reset()
                    self.agent.initialize(self.world.trajectory)
                    self.agent.reset()
                    variables.reset()
                    live.update(progress.layout)

                    done = False
                    score = 0
                    for itern in range(settings.MAX_EPISODE_LEN):
                        progress.update_task(t2id, completed=itern)

                        # get state  and  have agent get controls
                        state_idx = self.world.curr_traj_waypoint_idx
                        state = self.get_state()
                        controls = self.agent.act(state)
                        if controls is None:
                            progress.log(
                                "MANAGER", "Trial done because control is nan"
                            )
                            done = True
                            break

                        variables.itern = itern
                        variables.state = state
                        variables.action = controls
                        variables.traj_idx = self.world.curr_traj_waypoint_idx

                        state_live.update(self.agent, self.world)

                        # move the agent with the selected controls
                        self.agent.move(controls)

                        # get next state and reward
                        next_state = self.get_state()
                        dx, dy, reward = self.world.get_reward(
                            self.agent, state_idx
                        )
                        variables.error = (dx, dy)
                        if reward is None:
                            # logger.debug('Stopped because reward was NONE')
                            done = True
                            break

                        # check if we are done
                        if (
                            itern >= settings.MAX_EPISODE_LEN
                            or self.world.isdone(
                                self.agent, settings.MIN_GOAL_DISTANCE
                            )
                        ):
                            if itern == settings.MAX_EPISODE_LEN:
                                progress.log(
                                    "MANAGER",
                                    "Trial done because reached max number of episodes",
                                )
                            done = True

                        # train the agent
                        self.agent.step(
                            state, controls, reward, next_state, done
                        )

                        score += reward
                        variables.reward = reward
                        # time.sleep(1)

                        if done or h.interrupted:
                            break

                    scores.append(score)
                    progress.log("MANAGER", f"Score: {score:.2f}")

                    progress.remove_task(t2id)
                    progress.update_task(tid, completed=n + 1)
                    live.update(progress.layout)

                    if h.interrupted:
                        break

        plt.plot(scores[:-2])
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()
