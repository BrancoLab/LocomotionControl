# %%
from proj import Model, Environment, Controller, run_experiment, plot_trajectory
import matplotlib.pyplot as plt

# ! TODO  formula I_c for cube!
# TODO once it's mature enough make proper code testing stuff

agent = Model()
env = Environment(agent)
control = Controller(agent)

# plot_trajectory(env.reset())
# plt.show()

run_experiment(env, control, agent, n_steps=2000)



# # %%
# import matplotlib.pyplot as plt
# from matplotlib import transforms

# img = plt.imread('/Users/federicoclaudi/Desktop/mouse.png')

# fig = plt.figure()
# ax = fig.add_subplot(111)

# traj = env.reset()


# ax.plot(traj[:, 0], traj[:, 1])

# # transform to set to zero
# tr = transforms.Affine2D().scale(.05).rotate_deg(180).translate(10, 10)

# tr = tr.translate(agent.curr_x.x, agent.curr_x.y).rotate(agent.curr_x.theta)

# ax.imshow(img, transform=tr + ax.transData)
# _ = ax.set(xlim=[-10, 40], ylim=[-10, 110])