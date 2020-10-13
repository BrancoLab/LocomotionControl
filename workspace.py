# %%
from proj import Model

mod = Model()
mod.get_combined_dynamics_kinematics()
g, M, inp = mod.matrixes["g"], mod.matrixes["M"], mod.matrixes["inp"]

eq = g + M * inp
print(eq)
# %%
eq
# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.set(
    xlabel="# frames",
    ylabel="Torque\n($\\frac{cm^2 g}{s^2}$)",
    title="Control history",
)

# %%
