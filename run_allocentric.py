# %%

from proj import (
    Model,
    Environment,
    Controller,
    # run_experiment,
    # plot_trajectory,
    # run_manual,
)

# from proj.animation.animate import animate_from_images
from proj.utils import load_results_from_folder

import matplotlib.pyplot as plt
import numpy as np

# from fcutils.maths.utils import derivative


# ! TODO  formula I_c for cube!
# ! energy considerations, you should need more force to accelerate when going faster?

# TODO fix orientation at start of trajectory

agent = Model()
env = Environment(agent)
control = Controller(agent)

n_steps = 500

t = np.linspace(0, 1, n_steps)
u = np.ones((n_steps, 2)) * 2
u[:, 0] = (np.sin(t) - 0.5) * 2
# u[:, 1] = np.cos(t) * 2
agent.model

# %%
config, control, state, trajectory, history = load_results_from_folder(
    "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/parabola_200905_134410"
)

goal_x, goal_y = trajectory[-1, 0], trajectory[-1, 1]


f = plt.figure()
ax = f.add_subplot(121)
pax = f.add_subplot(122, projection="polar")


ax.scatter(goal_x, goal_y, s=100, color="r", marker="D")

gammas, rs, vs = [], [], []
for i in range(len(history["theta"][50:])):
    agent.curr_x = agent._state(
        history["x"][i + 50],
        history["y"][i + 50],
        history["theta"][i + 50],
        history["v"][i + 50],
        history["omega"][i + 50],
    )

    if i % 50 == 0:
        ax.scatter(agent.curr_x.x, agent.curr_x.y, color="k")

        ax.plot(
            [agent.curr_x.x, agent.curr_x.x + np.cos(agent.curr_x.theta) * 10],
            [agent.curr_x.y, agent.curr_x.y + np.sin(agent.curr_x.theta) * 10],
            lw=1,
            color=[0.4, 0.4, 0.4],
        )

        # ax.plot([agent.curr_x.x, goal_x],
        #             [agent.curr_x.y, goal_y], color='k', ls='--')

    r, gamma = agent.calc_angle_distance_from_goal(goal_x, goal_y)
    gammas.append(gamma)
    rs.append(r)
    vs.append(history["v"][50 + i])

pax.scatter(gammas, rs, c=np.arange(len(gammas)), cmap="bwr")

_ = ax.axis("equal")

# %%
f, ax = plt.subplots()
a, b = [], []
for n, gamma in enumerate(gammas):
    omega = history["omega"][50 + n] * agent.dt
    v = vs[n]
    r = rs[n]

    gamma_dot = gammas[n + 1] - gamma
    gamma_dot_calc = np.arcsin(
        (np.sin(gamma) * v)
        / (r ** 2 + v ** 2 - 2 * r * v * np.cos(gamma)) ** 0.5
    )
    gamma_dot_calc = gamma_dot_calc - (omega * -np.sign(gamma))
    gamma_dot_calc *= agent.dt

    a.append(gamma_dot)
    b.append(gamma_dot_calc)

    if n + 2 == len(gammas):
        break
    # break

ax.scatter(a, b, cmap="bwr", c=np.arange(len(b)), alpha=0.5)
_ = ax.axis("equal")


# %%
