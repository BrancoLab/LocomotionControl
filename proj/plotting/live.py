import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from fcutils.maths.utils import derivative
from fcutils.plotting.colors import desaturate_color

def plot_mouse(history, mouse, ax):
    """
        Given the state and mouse params plots a mouse
    """
    # plot body
    theta = np.degrees(history['theta'])[-1]
    ms = Arc((history['x'][-1], history['y'][-1]), mouse['length'], mouse['L']*2, 
                color='magenta',
                 angle=theta, linewidth=4, fill=False, zorder=2)

    # plot head
    ms2 = Arc((history['x'][-1], history['y'][-1]), mouse['length'], mouse['L']*2, color='g',
                    theta1 = theta-90, theta2 = theta+90,
                    angle=theta, linewidth=8, fill=False, zorder=2)

    ax.add_patch(ms)
    ax.add_patch(ms2)
  

def update_interactive_plot(axarr, model, goal, trajectory, g_xs, niter):
    """
        Update plot with current situation and controls
    """
    x = model.curr_x
    u = model.curr_control

    axarr[0].clear()
    axarr[1].clear()
    axarr[2].clear()
    axarr[3].clear()
    axarr[4].clear()
    axarr[5].clear()

    # plot trajectory
    axarr[0].scatter(trajectory[:, 0], trajectory[:, 1], 
                c=trajectory[:, 2], alpha=.8, zorder=-1)

    # plot currently used goal states
    axarr[0].plot(g_xs[:, 0], g_xs[:, 1], 
                lw=3, color='r', alpha=1, zorder=-1)

    # plot mouse and XY tracking history
    plot_mouse(model.history, model.mouse, axarr[0])
    axarr[0].plot(model.history['x'], model.history['y'], color='g', lw=1.5, ls='--')

    # update ax
    axarr[0].set(title=f'ITER: {niter} | x:{round(x.x, 2)}, y:{round(x.y, 2)}, ' +
                        f' theta:{round(np.degrees(x.theta), 2)}, v:{round(x.v, 2)}\n'+
                        f'GOAL: x:{round(goal.x, 2)}, y:{round(goal.y, 2)}, ' +
                        f' theta:{round(np.degrees(goal.theta), 2)}, v:{round(goal.v, 2)}',
                        # xlim=[-15, params['distance']+15], ylim=[-15, params['distance']+15],
                        )
    axarr[0].axis('equal')

    if len(model.history['omega']) > 5:
        # Plot Angular velocity
        axarr[1].plot(model.history['omega'][5:], color='m', lw=4, label='$\omega$')
        axarr[1].legend()
        axarr[1].set(title='Angular velocity')

        # Pot angular accelratopm
        axarr[2].plot(derivative(np.array(model.history['omega'][5:])), color='m', lw=4, label='$\dot{\omega}$')
        axarr[2].legend()
        axarr[2].set(title='Angular acceleration')

        # plot controls history
        axarr[3].plot(model.history['tau_l'][5:], color='b', lw=4, label='$\\tau_L$')
        axarr[3].plot(model.history['tau_r'][5:], color='r', lw=4, label='$\\tau_R$')
        axarr[3].legend()
        axarr[3].set(title='Control')

        # Plot linear velocity
        axarr[4].plot(model.history['v'][5:], color='g', lw=4, label='$v$')
        axarr[4].legend()
        axarr[4].set(title='Linear velocity')

        # Plot linear acceleration
        axarr[5].plot(derivative(np.array(model.history['v'][5:])), color=desaturate_color('g'), lw=4, label='$\dot{v}$')
        axarr[5].legend()
        axarr[5].set(title='Linear acceleration')