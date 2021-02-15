import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger

from rich.live import Live

from myterial import orange, salmon
from pyrnn._utils import GracefulInterruptHandler

from RL.agent import RLAgent
from RL.world import RLWorld
from RL import settings
from RL.progress import Progress, Variables, State, Trajectory, RewardPlot


class Manager:
    def __init__(self):
        self.agent = RLAgent()
        self.world = RLWorld()
        logger.remove()

    def get_inputs(self):
        """
            Gets the input to the agent at a given
            step in a simulation
        """
        # get the current trajectory input
        r, psy, v, omega = self.world.get_agent_trajectory_input(self.agent)

        # get the agent's state
        # v, omega = self.agent.curr_x.v, self.agent.curr_x.omega

        return np.array([r, psy, v, omega])

    def save(self):
        torch.save(self.agent.actor_local.state_dict(), "checkpoint_actor.pth")
        torch.save(
            self.agent.critic_local.state_dict(), "checkpoint_critic.pth"
        )

    def train(self):
        rewardplot = RewardPlot()
        trajectory_record = Trajectory()
        variables = Variables()
        state_live = State()
        progress = Progress(self.agent, variables, state_live, rewardplot)
        progress.log("START", "Starting trainig")
        tid = progress.add_task(
            description=f"[{salmon}]Training", total=settings.N_EPISODES
        )

        self.world.progress = progress

        scores = []
        rewardplot.scores = scores
        with Live(progress.layout, screen=True, refresh_per_second=60) as live:
            with GracefulInterruptHandler() as h:
                for n in range(settings.N_EPISODES):
                    t2id = progress.add_task(
                        description=f"[{orange}]Trial {n+1}",
                        total=settings.MAX_EPISODE_LEN,
                    )

                    self.world.reset()
                    trajectory_record.trajectory = self.world.trajectory

                    self.agent.initialize(self.world.trajectory)
                    variables.reset()
                    variables.traj_len = len(self.world.trajectory)
                    live.update(progress.layout)

                    done = False
                    score = 0
                    for itern in range(settings.MAX_EPISODE_LEN):
                        progress.update_task(t2id, completed=itern)

                        # get state  and  have agent get controls
                        state_idx = self.world.curr_traj_waypoint_idx
                        try:
                            inputs = self.get_inputs()
                        except IndexError:
                            progress.log(
                                "MANAGER",
                                "seems like we reached the end of the trajectory",
                            )
                            done = True
                            break

                        controls = self.agent.act(inputs)
                        if controls is None:
                            progress.log(
                                "MANAGER", "Trial done because control is nan"
                            )
                            done = True
                            break

                        # set variables for display
                        variables.itern = itern
                        variables.memory_len = len(self.agent.memory)
                        variables.inputs = {
                            k: v
                            for k, v in zip(("r", "psy", "v", "omega"), inputs)
                        }
                        variables.traj_idx = self.world.curr_traj_waypoint_idx
                        variables.score = score
                        if score > variables.max_score:
                            variables.max_score = score
                        state_live.update(self.agent, self.world, controls)

                        # move the agent with the selected controls
                        self.agent.move(controls)

                        # get next state and reward
                        next_state = self.get_inputs()
                        dx, dy, reward = self.world.get_reward(
                            self.agent, state_idx, controls
                        )
                        variables.error = (dx, dy)
                        if reward is None:
                            progress.log("MANAGER", "Reward is NONE, stopping")
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

                        # add to memory
                        self.agent.memory.add(
                            inputs, controls, reward, next_state, done
                        )

                        # train the agent
                        self.agent.step()

                        # keep track of score
                        score += reward
                        variables.reward = reward

                        if score >= 0.95:
                            progress.log("MANAGER", "Done learnin")
                            h.interrupted = True
                            done = True
                            break

                        if done or h.interrupted:
                            progress.log("MANAGER", "done!")
                            break

                    # save
                    if n % 100 == 0:
                        progress.log("MANAGER", f"Saving models at iter: {n}")
                        self.save()

                    # keep track of things
                    scores.append(score)
                    progress.log("MANAGER", f"Score: {score:.2f}")

                    if score == 0.0:
                        progress.log(
                            "MANAGER",
                            "For some reason things crashed and the score is 0",
                        )
                        # raise ValueError('For some reason things crashed and the score is 0')

                    progress.remove_task(t2id)
                    progress.update_task(tid, completed=n + 1)
                    live.update(progress.layout)

                    if h.interrupted:
                        break

        plt.plot(scores[:-2])
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()
