import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger

# import time

from rich.live import Live

from myterial import orange, salmon
from pyrnn._utils import GracefulInterruptHandler

from RL.agent import RLAgent
from RL.world import RLWorld
from RL import settings
from RL.progress import Progress


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
        try:
            r, psy, v, omega = self.world.get_agent_trajectory_input(
                self.agent
            )
        except Exception:
            return None

        # scale values
        # see https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range/281164
        to_scale = dict(
            r=(r, settings.R_MAX),
            psy=(psy, settings.PSY_MAX),
            v=(v, settings.V_MAX),
            omega=(omega, settings.OMEGA_MAX),
        )
        scaled = {}
        for name, (val, limit) in to_scale.items():
            if val > limit.max:
                self.progress.log(
                    "MANAGER",
                    f"[b{ salmon}]Variable {name} was above limit {round(val, 2)}!!",
                )
                return None
            elif val < limit.min:
                self.progress.log(
                    "MANAGER",
                    f"[b{ salmon}]Variable {name} was below limit {round(val, 2)}!!!",
                )
                return None
            scaled[name] = (val - limit.min) / (limit.max - limit.min) * 2 - 1

        return np.array(list(scaled.values()))

    def save(self, name=""):
        torch.save(
            self.agent.actor_local.state_dict(), f"checkpoint_actor_{name}.pth"
        )
        torch.save(
            self.agent.critic_local.state_dict(),
            f"checkpoint_critic_{name}c.pth",
        )

    def setup_training(self):
        self.progress = Progress()
        self.world.progress = self.progress
        self.progress.log("START", "Starting trainig")

        self.progress.add_task(
            description=f"[{salmon}]Training",
            total=settings.N_EPISODES,
            _name="main",
        )
        self.progress.add_task(
            description=f"[{orange}]Trial {0}",
            total=settings.MAX_EPISODE_LEN,
            _name="trial",
        )

    def initialize_episode(self, n):
        self.world.reset()
        self.agent.initialize(self.world.trajectory)
        self.progress.variables.traj_len = len(self.world.trajectory)
        self.progress.reset()

    def train(self):
        self.setup_training()
        scores = []
        # Handle LIVE during training
        with Live(self.progress.layout, screen=True, refresh_per_second=20):
            with GracefulInterruptHandler() as h:

                # Loop over EPISODES
                for n in range(settings.N_EPISODES):
                    self.initialize_episode(n)
                    self.progress.log("NEW TRIAL", "=" * 10)
                    # live.update(self.progress.layout, refresh=True)
                    # time.sleep(3)

                    # Loop over FRAMES
                    done, score = False, 0
                    for itern in range(settings.MAX_EPISODE_LEN):
                        self.progress.update_task("trial", completed=itern)

                        # get agent inputs
                        INITIAL_STATE_INDEX = self.world.curr_traj_waypoint_idx
                        INPUTS = self.get_inputs()
                        if INPUTS is None:
                            self.progress.log("MANAGER", "Out of bound inputs")
                            done = True
                            break

                        CONTROLS = self.agent.get_controls(INPUTS)
                        if CONTROLS is None:
                            self.progress.log(
                                "MANAGER", "Done because control is None"
                            )
                            done = True
                            break

                        # move the agent with the selected controls
                        self.agent.move(CONTROLS)

                        # get next state and reward
                        NEXT_STATE = self.get_inputs()
                        if NEXT_STATE is None:
                            self.progress.log("MANAGER", "Out of bound inputs")
                            done = True
                            break

                        reward = self.world.get_reward()
                        score += reward

                        # Update variables for live displayu
                        self.progress.update(
                            itern,
                            len(self.agent.memory),
                            INPUTS,
                            INITIAL_STATE_INDEX,
                            reward,
                            score,
                            scores,
                            self.agent,
                            self.world,
                            CONTROLS,
                        )

                        # check if we are done
                        if (
                            itern >= settings.MAX_EPISODE_LEN
                            or self.world.isdone(
                                self.agent, settings.MIN_GOAL_DISTANCE
                            )
                        ):
                            if itern == settings.MAX_EPISODE_LEN:
                                self.progress.log(
                                    "MANAGER",
                                    "Trial done because reached max number of episodes",
                                )
                            done = True

                        # add to memory
                        self.agent.memory.add(
                            INPUTS, CONTROLS, reward, NEXT_STATE, done
                        )

                        # train the agent
                        self.agent.fit()

                        if done or h.interrupted:
                            self.progress.log("MANAGER", "done!")
                            break

                    # ### end of trial loop ###
                    self.progress.log(
                        "MANAGER", f"Finished EPISODE after {itern} FRAMES"
                    )

                    # save
                    if n % 100 == 0:
                        self.progress.log(
                            "MANAGER", f"Saving models at iter: {n}"
                        )
                        self.save()

                    # keep track of things
                    scores.append(score)
                    self.progress.log("MANAGER", f"Score: {score:.2f}")

                    if score == 0.0:
                        self.progress.log(
                            "MANAGER",
                            "For some reason things crashed and the score is 0",
                        )

                    self.progress.update_task(
                        "trial", description=f"[b {orange}]Trial {n+1}"
                    )
                    self.progress.update_task("main", completed=n + 1)
                    if h.interrupted:
                        break

        self.save(name="final")
        plt.plot(scores[:-2])
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()
