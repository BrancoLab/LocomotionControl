# %%
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import control

from fcutils.maths.geometry import calc_distance_between_points_2d as points_distance
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as points_angle


# %%
def compute_position_from_state(x, goal=None):
    # Compute agents position in allocentric coordinates given state
    x0 = np.sin(np.radians(x.theta))*x.d 
    x1 = np.cos(np.radians(x.theta))*x.d

    if goal is not None:
        x0 = goal.x0 - x0
        x1 = goal.x1 - x1

    return point(x0, x1)

def plot_results(trace, states, controls):
    try:
        arrived = np.where(np.array([s.d for s in states]) < 1)[0][0]
    except:
        arrived = 'NO'

    f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    axarr=axarr.flatten()

    axarr[0].scatter([p.x0 for p in trace], [p.x1 for p in trace], 
                        c = np.arange(len(trace)),
                        s=30, cmap='bwr')
    axarr[0].set(title=f'Arrived at: {arrived}')

    axarr[1].plot([s.d for s in states], label='d')
    axarr[1].plot([s.theta for s in states], label='theta')
    axarr[1].plot([s.L for s in states], label='L speed')
    axarr[1].plot([s.R for s in states], label='R speed')
    axarr[1].legend()

    axarr[2].plot([c[0] for c in controls], label='L accel')
    axarr[2].plot([c[1] for c in controls], label='R accel')
    axarr[2].legend()

    axarr[3].plot([s.d for s in states], [s.theta for s in states])
    axarr[3].set(xlabel='d', ylabel='theta')

def run(x0):
    K, S, E = control.lqr(A, B, Q, R)

    # Simulation params
    steps = 2000

    # Set initial state
    x = x0 # initial state, note theta = 0 means right north of the goal

    # Keep track of where the agent has been
    trace = [compute_position_from_state(x)]
    states = [x]
    controls = []

    for step in range(steps):
        controls.append(np.array(- K @ np.array(x)).ravel())
        x_dot = state(*np.array((A - B @ K) @ np.array(x)).ravel() * dt)

        x = state(x.d + x_dot.d, 
                    x.theta + x_dot.theta, 
                    x.L + x_dot.L, 
                    x.R + x_dot.R, 
                )
        
        trace.append(compute_position_from_state(x))
        states.append(x)

    # Plot results
    plot_results(trace, states, controls)

    return trace, states, controls

# %%
state = namedtuple('state', 'd, theta, L, R')
point = namedtuple('point', 'x0, x1')
dt = 0.002

mu = 0

A = np.array([
    [0, 1, -1, -1],
    [0, 0, -1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

B = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
])


Q = np.array([
    [1, 0, 0, 0],
    [0, .01, 0, 0],
    [0, 0, 0.01, 0],
    [0, 0, 0, 0.01],
])


R = np.array([
    [1, 0],
    [0, 1],
])


t, s, c  = run(state(80, 5, 0, 0))

# %%
# --------------------------------- waypoints -------------------------------- #

K, S, E = control.lqr(A, B, Q, R)

def get_d_theta_given_pos_and_goal(pos, goal):
    d = points_distance(np.array(pos), np.array(goal))
    theta = points_angle(pos.x0, pos.x1, goal.x0, goal.x1)

    return d, theta

steps = 5000

# define three points the agent needs to reach 
w1 = point(10, 0)
w2 = point(20, 50)
w3 = point(100, 100)

waypoints = [w1, w2, w3]

# define the initial position of the agent, and initial goal
pos = point(20, 0)
goal = 0

# get initial state
d, t = get_d_theta_given_pos_and_goal(pos, waypoints[goal])
x = state(d, t, 0, 0)

trace = [compute_position_from_state(x, goal=waypoints[goal])]
states = [x]
switches = []
controls = []
for i in range(steps):
    controls.append(np.array(- K @ np.array(x)).ravel())

    # step
    x_dot = state(*np.array((A - B @ K) @ np.array(x)).ravel() * dt)

    x = state(x.d + x_dot.d, 
                x.theta + x_dot.theta, 
                x.L + x_dot.L, 
                x.R + x_dot.R, 
            )

    # if i == 1200: break

    if x.d <= 5:
        print('switching goal ', i)
        goal += 1
        if goal >= len(waypoints): 
            break
        switches.append(i)
        

        pos = compute_position_from_state(x, goal=waypoints[goal])
        d, t = get_d_theta_given_pos_and_goal(pos, waypoints[goal])
        
        if t > 180:
            t = 360- t
        x = state(d, t, x.L, x.R)

    trace.append(compute_position_from_state(x, goal=waypoints[goal]))
    states.append(x)

    if i == 1687: break



f, axarr = plt.subplots(figsize=(12, 8), ncols=3)
for p in waypoints:
    axarr[0].scatter(p.x0, p.x1)

axarr[0].scatter(trace[0].x0, trace[0].x1, marker='*', s=50)
axarr[0].plot([p.x0 for p in trace], [p.x1 for p in trace])


axarr[1].plot([s.d for s in states], label='d')
axarr[1].plot([s.theta for s in states], label='theta')
axarr[1].plot([s.L for s in states], label='L speed')
axarr[1].plot([s.R for s in states], label='R speed')


axarr[2].plot([c[0] for c in controls], label='L accel')
axarr[2].plot([c[1] for c in controls], label='R accel')
axarr[2].legend()

for sw in switches:
    axarr[1].axvline(sw)

axarr[1].legend()

# axarr[0].set(xlim=[-5, 105], ylim=[-5, 105])

# %%
