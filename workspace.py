# %%
from proj import Model
from sympy import Matrix, symbols, sin, cos

model = Model()

# %%
(
    x,
    y,
    theta,
    L,
    R,
    m,
    m_w,
    d,
    tau_l,
    tau_r,
    v,
    omega,
) = model.variables.values()

# %%
x_dot, y_dot, theta_dot = symbols("xdot, ydot, thetadot")

nu_l_dot, nu_r_dot = symbols("nudot_L, nudot_R")


vels = Matrix([x_dot, y_dot, theta_dot])

K = Matrix(
    [
        [R / 2 * cos(theta), R / 2 * cos(theta)],
        [R / 2 * sin(theta), R / 2 * sin(theta)],
        [R / (2 * L), R / (2 * L)],
    ]
)

nu = Matrix([nu_l_dot, nu_r_dot])

nu = K.pinv() * vels
nu
# %%
