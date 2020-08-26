import numpy as np

"""
    Interactive plot during running
"""

def interactive_plot(axarr, x, goal, u, info, g_xs):
    axarr[0].clear()
    axarr[1].clear()
    axarr[0].scatter(info['goal_state'][:, 0], info['goal_state'][:, 1], 
                color='g', alpha=.2, zorder=-1)

    axarr[0].plot(g_xs[:, 0], g_xs[:, 1], 
                lw=3, color='k', alpha=1, zorder=-1)

    axarr[0].scatter(x.x, x.y, s=160, c=x.v, vmin=-10, vmax=10, cmap='bwr', 
                            lw=2, edgecolors='k')

    axarr[0].plot([x.x, x.x + np.cos(x.theta) * .5],
                    [x.y, x.y + np.sin(x.theta) * .5],
                    lw=2, color='r', zorder=99)
    axarr[0].set(title=f'ITER: {iter} | x:{round(x.x, 2)}, y:{round(x.y, 2)}, ' +
                        f' theta:{round(np.degrees(x.theta), 2)}, v:{round(x.v, 2)}\n'+
                        f'GOAL: x:{round(goal.x, 2)}, y:{round(goal.y, 2)}, ' +
                        f' theta:{round(np.degrees(goal.theta), 2)}, v:{round(goal.v, 2)}',
                        # xlim=[-1.7, 1.7], ylim=[-.1, 2.1],
                        )
    axarr[1].bar([0, 1], u, color=['b', 'r'])
    axarr[1].set(title='control', xticks=[0, 1], xticklabels=['L', 'R'])