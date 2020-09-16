# %%
from proj import Model

model = Model()
model.get_combined_dynamics_kinematics()
model.get_jacobians()
# %%

"""
    model.model is the matrix representing the ODEs system

    dxdt = MODEL

"""

model.model.simplify()
model.model

# %%
