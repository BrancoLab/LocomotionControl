# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from scipy.optimize import curve_fit
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d

# %%





# %%
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

angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))
speed = 1 - np.sin(np.linspace(0, 3, len(x)))


f, ax = plt.subplots(figsize=(12, 12))
ax.scatter(x, y, c=speed, cmap='bwr')

# draw mouse


ax.set(xlim=[-15, params['distance']+15], ylim=[-15, params['distance']+15])


# %%
