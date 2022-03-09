import numpy as np

# Vehicle config
l_f = 4  # [cm]
l_r = 2  # [cm]

wheelbase = l_f + l_r  # wheel base: front to rear axle [cm]
wheeldist = 3  # wheel dist: left to right wheel [cm]

# for plotting the car
v_w = 4  # vehicle width [cm]
r_b = 1  # rear to back [cm]
r_f = 7  # rear to front [cm]
t_r = 0.40  # tire radius [cm]
t_w = 0.30  # tire width [cm]


# Controller Config
dt = 0.02  # [s]
max_iteration = 250
eps = 0.05

matrix_q = [2.0, 0.0, 1.0, 0.0]
matrix_r = [0.5]

state_size = 4

max_acceleration = 50.0  # [cm / s^2]
max_steer_angle = np.deg2rad(80)  # [rad]
max_speed = 100 / 3.6  # [km/h]

# ------------------------------ lon PID ------------------------------ #
longitudinal_pid_P_gain = 0.3
longitudinal_pid_D_gain = 0.5
longitudinal_pid_I_gain = 0.1


# ----------------------------------- pure ----------------------------------- #
look_ahead_distance = 0.1
look_ahead_gain = 3
