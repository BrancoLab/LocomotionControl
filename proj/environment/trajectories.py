import numpy as np
from scipy.optimize import curve_fit
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d


def complete_given_xy(x, y, params):
    # Compute other variables that figure in the state vector
    angle = np.radians(90 - calc_angle_between_points_of_vector_2d(x, y))
    angle = np.unwrap(angle)

    speed = (1 - np.sin(np.linspace(0, 3, len(x)))) 
    speed = speed * (params['max_speed']-params['min_speed']) + params['min_speed']
    
    ang_speed = np.ones_like(speed) # it will be ignored


    trajectory = np.vstack([x, y, angle, speed, ang_speed]).T
    return trajectory


# ---------------------------------- Curves ---------------------------------- #
def line(n_steps, params):
    y = np.linspace(0, params['distance'], n_steps)
    x = np.zeros_like(y)

    return complete_given_xy(x, y, params)

def circle(n_steps, params):
    p = np.linspace(0, 2 * np.pi, n_steps)
    r = params['distance']/2

    x = np.cos(p) * r
    y = np.sin(p) * r

    return complete_given_xy(x, y, params)


def sin(n_steps, params):
    x = np.linspace(0, params['distance'], n_steps)
    y = 5 * np.sin(.1 * x)

    return complete_given_xy(x, y, params)

def parabola(n_steps, params):
    def curve(x, a, b, c):
        return (a * (x-b)**2) + + c

    # Define 3 points
    X = [0, params['distance']/2, params['distance']]
    Y = [0, params['distance']/4, 0]

    # fit curve and make trace
    coef,_ = curve_fit(curve, X, Y)

    y = np.linspace(0, params['distance'], n_steps)
    x = curve(y, *coef)

    return complete_given_xy(x, y, params)