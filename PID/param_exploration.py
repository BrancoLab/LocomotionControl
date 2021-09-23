import sys

sys.path.append("./")


from PID.simulation import Simulation
from PID.pid import PID


params = dict(
    proportional_gain=[8, 8, 8],
    integral_gain=[0, 1e-8, 1e-6, 1e-2, 1],
    derivative_gain=[0, 1e-8, 1e-4, 1e-3, 1],
)

for prop in params["proportional_gain"]:
    for intgrl in params["integral_gain"]:
        for der in params["derivative_gain"]:
            prms = dict(
                proportional_gain=prop,
                integral_gain=intgrl,
                derivative_gain=der,
            )
            print(f"Running with params:\n{prms}")

            # initialize controllers
            angle_pid = PID(
                proportional_gain=prop,
                integral_gain=intgrl,
                derivative_gain=der,
                dt=0.01,
            )

            speed_pid = PID(1e-2, 0, 1e-4, dt=0.01)

            sim = Simulation(angle_pid, speed_pid)
            sim.run()
            sim.visualize()
            del sim.history
            break
        break
