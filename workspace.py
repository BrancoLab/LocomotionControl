from proj import Model
from rich import print
from sympy import latex
from pylatexenc.latex2text import LatexNodes2Text

model = Model()
model.get_combined_dynamics_kinematics()

g, inp, M = model.matrixes["g"], model.matrixes["inp"], model.matrixes["M"]

mod = g + M * inp

print(LatexNodes2Text().latex_to_text((latex(inp))))
