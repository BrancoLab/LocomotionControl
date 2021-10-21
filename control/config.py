import numpy as np

# Vehicle config
l_f = 1.165  # [m]
l_r = 1.165  # [m]

wheelbase = l_f + l_r  # wheel base: front to rear axle [m]
wheeldist = 1.85  # wheel dist: left to right wheel [m]
v_w = 2.33  # vehicle width [m]
r_b = 0.80  # rear to back [m]
r_f = 3.15  # rear to front [m]
t_r = 0.40  # tire radius [m]
t_w = 0.30  # tire width [m]

c_f = 155494.663  # [N / rad]
c_r = 155494.663  # [N / rad]
m_f = 570  # [kg]
m_r = 570  # [kg]

Iz = 1436.24  # [kg m2]

# Controller Config
dt = 0.10  # [s]
max_iteration = 250
eps = 0.01

matrix_q = [1.0, 0.0, 1.0, 0.0]
matrix_r = [1.0]

state_size = 4

max_acceleration = 5.0  # [m / s^2]
max_steer_angle = np.deg2rad(40)  # [rad]
max_speed = 30 / 3.6  # [km/h]
