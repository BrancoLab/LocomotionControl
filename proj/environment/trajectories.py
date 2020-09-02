import numpy as np
from scipy.optimize import curve_fit
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d


def parabola(n_steps, params):
    def curve(x, a, b, c):
        return (a * (x-b)**2) + + c

    # Define 3 points
    X = [0, params['distance']/2, params['distance']]
    Y = [0, params['distance']/4, 0]

    # fit curve and make trace
    coef,_ = curve_fit(curve, X, Y)

    x = np.linspace(0, params['distance'], n_steps)
    y = curve(x, *coef)

    # Compute other variables that figure in the state vector
    angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))

    speed = (1 - np.sin(np.linspace(0, 3, len(x)))) 
    speed = speed * (params['max_speed']-params['min_speed']) + params['min_speed']

    ang_speed = np.ones_like(speed) # it will be ignored


    trajectory = np.vstack([x, y, angle, speed, ang_speed]).T
    return trajectory[1:, :]