# %%
"""
    Simulation of control strategy for an agent that can move in 2d and needs to
    reach a goal location. The agent's state is defined by its distance (d) 
    and angle (theta) from the goal and d_dot, theta_dot. 

    The control can affect d_dot and theta_dot and is expressed as 
    d_dotdot and theta_dotdot.
"""

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

def plot_results(trace, goal_location, states):
    try:
        arrived = np.where(np.array([s.d for s in states]) < 1)[0][0]
    except:
        arrived = 'NO'

    f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    axarr=axarr.flatten()

    axarr[0].scatter(goal_location.x0, goal_location.x1, s=50, c='r')
    axarr[0].scatter([p.x0 for p in trace], [p.x1 for p in trace], 
                        c = np.arange(len(trace)),
                        s=30, cmap='bwr')
    axarr[0].scatter(trace[0].x0, trace[0].x1, s=100, c='b')
    axarr[0].set(title=f'Arrived at: {arrived}')

    axarr[1].plot([s.d for s in states], label='d')
    axarr[1].plot([s.d_dot for s in states], label='d_dot')
    axarr[2].plot([s.theta for s in states], label='theta')
    axarr[2].plot([s.theta_dot for s in states], label='theta_dot')
    axarr[1].legend()
    axarr[2].legend()

    axarr[3].plot([s.d for s in states], [s.theta for s in states])
    axarr[3].scatter([s.d for s in states][0], [s.theta for s in states][0], s=20)
    axarr[3].set(xlabel='d', ylabel='theta')

    axarr[0].set(xlim=[-80, 80], ylim=[-25, 80])


# %%

def run(x0):
    K, S, E = control.lqr(A, B, Q, R)


    # Useful stuff
    state = namedtuple('state', 'd, theta, d_dot, theta_dot')
    point = namedtuple('point', 'x0, x1')

    # Simulation params
    steps = 5000
    dt = 0.001

    # Set initial state
    x = x0 # initial state, note theta = 0 means right north of the goal

    # Set goal location
    goal_location = point(0, 0)

    # Keep track of where the agent has been
    trace = [compute_position_from_state(x)]
    states = [x]

    for step in range(steps):
        u = - K @ np.array(x)
        x_dot = state(*np.array((A - B @ K) @ np.array(x)).ravel() * dt)

        x = state(x.d + x_dot.d, 
                    x.theta + x_dot.theta, 
                    x.d_dot + x_dot.d_dot, 
                    x.theta_dot + x_dot.theta_dot)
        
        trace.append(compute_position_from_state(x))
        states.append(x)

    # Plot results
    plot_results(trace, goal_location, states)



# %%
"""

The evolution of the system is defined by
x_dot = Ax + Bu

where:
"""

A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -10, 0, 10],
    [0, 0, 0, 0]
])

B = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
])


"""
And we use LQR to compute the best {u_0, ..., u_t}
given Q and R:
"""

Q = np.array([
    [10, 0, 0, 0],
    [0, .01, 0, 0],
    [0, 0, .01, 0],
    [0, 0, 0, .01]
]) 

R = np.array([
    [.1, 0],
    [0, 1000],
]) 


# RUN given an initial state
run(state(80, 45, 0, 0))

# TODO this  is all wrong because you can't be decreasing d when you're point away from the goal




# %%

# TODO make it work with waypoints
# update lqr
K, S, E = control.lqr(A, B, Q, R)

def get_d_theta_given_pos_and_goal(pos, goal):
    d = points_distance(np.array(pos), np.array(goal))
    theta = points_angle(pos.x0, pos.x1, goal.x0, goal.x1)

    return d, theta


# define three points the agent needs to reach 
w1 = point(10, 0)
w2 = point(10, 50)
w3 = point(100, 100)

waypoints = [w1, w2, w3]

# define the initial position of the agent, and initial goal
pos = point(20, 20)
goal = 0

# get initial state
d, t = get_d_theta_given_pos_and_goal(pos, waypoints[goal])
x = state(d, t, 0, 0)

trace = [compute_position_from_state(x, goal=waypoints[goal])]
for i in range(steps):
    # step
    x_dot = state(*np.array((A - B @ K) @ np.array(x)).ravel() * dt)

    x = state(x.d + x_dot.d, 
                x.theta + x_dot.theta, 
                x.d_dot + x_dot.d_dot, 
                x.theta_dot + x_dot.theta_dot)

    if x.d <= 5:
        print('switching goal')
        goal += 1
        if goal >= len(waypoints): 
            break

        pos = compute_position_from_state(x, goal=waypoints[goal])
        d, t = get_d_theta_given_pos_and_goal(pos, waypoints[goal])
        x = state(d, t, x.d_dot, x.theta_dot)

    trace.append(compute_position_from_state(x, goal=waypoints[goal]))


f, ax = plt.subplots()
for p in waypoints:
    ax.scatter(p.x0, p.x1)

ax.scatter(trace[0].x0, trace[0].x1, marker='*', s=50)

ax.plot([p.x0 for p in trace], [p.x1 for p in trace])

# ax.set(xlim=[-5, 105], ylim=[-5, 105])

# %%
