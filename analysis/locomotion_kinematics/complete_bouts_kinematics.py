# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
from loguru import logger


import kino.draw.locomotion as draw_locomotion


from analysis.load import load_complete_bouts
import draw

logger.remove()
logger.add(sys.stdout, level="INFO")

"""
    Plots complete bouts through the arena,
    but with the linearized track also
"""


# load and clean complete bouts
bouts = load_complete_bouts(keep=10, duration_max=12)


# %%

f, ax = plt.subplots(figsize=(7, 9))

draw.Hairpin(alpha=0.25)
# plot paws/com trajectory
draw_locomotion.plot_locomotion_2D(bouts[1], ax)

plt.show()


# %%
