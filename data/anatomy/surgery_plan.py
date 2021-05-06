# from probeplanner import Planner, Viewer
from brainrender import Scene
from brainrender.actors import ruler, Point
import numpy as np

# To plan probe placement
# planner = Planner('plan.yaml', 'myprobe.yaml')
# planner.plan()


# To measure distances and angles

BREGMA = np.array([5400, 0, 0])  # AP  # DV  # ML


top = np.array([4.136, -2.106, 0]) * 1000 + BREGMA  # AP ML DV
tip = np.array([5.507, -0.584, 6.489]) * 1000 + BREGMA


scene = Scene()
cun, grn = scene.add_brain_region("CUN", "GRN", alpha=0.4)

tip[1] = tip[1] + scene.root.centerOfMass()[2]
top[1] = top[1] + scene.root.centerOfMass()[2]

top = top[[0, 2, 1]]
tip = tip[[0, 2, 1]]

print(f"Distance between tip and top: {np.linalg.norm(top-tip):.3f}")

scene.add(Point(top))
scene.add(Point(tip))

p = cun.centerOfMass()
p[2] -= 1400
scene.add(Point(p))

rul1 = ruler(tip, top, unit_scale=0.001, units="mm")
rul2 = ruler(tip, p, unit_scale=0.001, units="mm")
rul3 = ruler(p, top, unit_scale=0.001, units="mm")
scene.add(rul2, rul3)

scene.render()
