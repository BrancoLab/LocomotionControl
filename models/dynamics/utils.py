import numpy as np
from scipy.optimize import curve_fit
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d
from matplotlib.patches import Arc



"""
    Interactive plot during running
"""
def make_road(params):
    """
        Defines a path that the mouse has to 
        follow, given a dictionary of params.
        Path is shaped as a parabola
    """
    def curve(x, a, b, c):
        return (a * (x-b)**2) + + c

    # Make road
    n_steps = int(params['duration']/params['dt'])

    # Define 3 points
    X = [0, params['distance']/2, params['distance']]
    Y = [0, params['distance']/4, 0]

    # fit curve and make trace
    coef,_ = curve_fit(curve, X, Y)

    x = np.linspace(0, params['distance'], n_steps)
    y = curve(x, *coef)
    # y = x 

    angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))

    speed = (1 - np.sin(np.linspace(0, 3, len(x)))) 
    speed = speed * (params['max_speed']-params['min_speed']) + params['min_speed']

    road = np.vstack([x, y, angle, speed]).T
    return road

def plot_mouse(x, mouse, ax, params):
    """
        Given the state and mouse params plots a mouse
    """
    # plot body
    theta = np.degrees(x.theta)
    ms = Arc((x.x, x.y), mouse['length'], mouse['L']*2, color='magenta',
                 angle=theta, linewidth=4, fill=False, zorder=2)

    # plot head
    ms2 = Arc((x.x, x.y), mouse['length'], mouse['L']*2, color='g',
                    theta1 = theta-30, theta2 = theta+30,
                    angle=theta, linewidth=8, fill=False, zorder=2)

    ax.add_patch(ms)
    ax.add_patch(ms2)
  

def interactive_plot(axarr, env, goal, info, g_xs, iter, mouse, params):
    """
        Update plot with current situation and controls
    """
    x = env.history_x[-1]
    u = env.history_u[-1]
    nu = env.history_nu[-1]

    axarr[0].clear()
    axarr[1].clear()
    axarr[2].clear()
    axarr[3].clear()

    # plot goal states
    axarr[0].scatter(info['goal_state'][:, 0], info['goal_state'][:, 1], 
                c=info['goal_state'][:, 2], alpha=.8, zorder=-1)

    # plot currently used goal states
    axarr[0].plot(g_xs[:, 0], g_xs[:, 1], 
                lw=3, color='r', alpha=1, zorder=-1)

    # plot mouse
    plot_mouse(x, mouse, axarr[0], params)

    x_hist = [x.x for x in env.history_x]
    y_hist = [x.y for x in env.history_x]
    axarr[0].plot(x_hist, y_hist, color='g', lw=1.5, ls='--')

    # update ax
    axarr[0].set(title=f'ITER: {iter} | x:{round(x.x, 2)}, y:{round(x.y, 2)}, ' +
                        f' theta:{round(np.degrees(x.theta), 2)}, v:{round(x.s, 2)}\n'+
                        f'GOAL: x:{round(goal.x, 2)}, y:{round(goal.y, 2)}, ' +
                        f' theta:{round(np.degrees(goal.theta), 2)}, v:{round(goal.s, 2)}',
                        xlim=[-15, params['distance']+15], ylim=[-15, params['distance']+15],
                        )
    axarr[0].axis('equal')

    # Plot controls
    axarr[1].bar([0, 1, 2], [u.L, u.R, u.L - u.R], color=['b', 'r', 'k'])
    axarr[1].set(title='control', xticks=[0, 1, 2], xticklabels=['L', 'R', 'L-R'])

    # plot controls history
    axarr[2].plot([u.L for u in  env.history_u], color='b', lw=4, label='L')
    axarr[2].plot([u.R for u in  env.history_u], color='r', lw=4, label='R')
    axarr[2].legend()
    axarr[2].set(title='Control')

    axarr[3].plot([nu.L for nu in  env.history_nu], color='g', lw=4, label='L')
    axarr[3].plot([nu.R for nu in  env.history_nu], color='m', lw=4, label='R')
    axarr[3].legend()
    axarr[3].set(title='Wheel speeds')