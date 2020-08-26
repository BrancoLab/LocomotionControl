A simple model with state

x = [x, y, theta, v].T

and dynamics:

x_dot = [
    v * cos(theta)
    v * sin(theta)
    u_theta
    u_v
]

and control u = [u_theat, u_V]