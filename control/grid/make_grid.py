import numpy as np
from pathlib import Path
from loguru import logger
from datetime import timedelta
import sys
from rich.progress import track

sys.path.append("./")
from control import config
from control.utils import to_json
from control import paths


logger.remove()
logger.add(sys.stdout, level="DEBUG")

"""
    Do a grid search of parameters for the control algorithm and save them 
    as .json file for running on HPC
"""

BASH = """#! /bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 1gb # memory pool for all cores
#SBATCH --job-name="GRID"
#SBATCH -n 1
#SBATCH --time=01:00:00
#SBATCH -o out.out
#SBATCH -e err.err

echo "loading conda env"
module load miniconda

conda activate locomotion

echo "Updating locomotion repobash  "
cd LocomotionControl

echo "locomoting"
python control/grid/run_one.py --config CONFIG_PATH
"""


def clean(params):
    """
        Removes np.ndarray from numpy arrays
    """
    params = params.copy()
    params["CONTROL_CONFIG"] = {
        k: int(v) if isinstance(v, int) else [float(V) for V in np.diag(v)]
        for k, v in params["CONTROL_CONFIG"].items()
    }
    return params


class Grid:
    def __init__(self, save_path):
        logger.debug(f"Saving GRID json files at: {save_path}")
        self.save_path = save_path
        # set a bunch of parameters and their value

        self.ranges = dict(
            R_run=(1e-3, 1e-2, 1e-1,),
            planning=[40, 60],
            Z_run=(1, 1e2, 1e3),
            Q_0=(30, 50, 100),  # x
            Q_1=(30, 50, 100),  # y
            Q_3=(10, 30, 50, 100),  # v
            Q_4=(10, 30, 50, 100),  # omega
        )

        self._params = dict(
            MANAGER_CONFIG=config.MANAGER_CONFIG,
            TRAJECTORY_CONFIG=config.TRAJECTORY_CONFIG,
            MOUSE=config.MOUSE,
            PLANNING_CONFIG=config.PLANNING_CONFIG,
            iLQR_CONFIG=config.iLQR_CONFIG,
            CONTROL_CONFIG=config.CONTROL_CONFIG,
            PARAMS=config.PARAMS,
        )

        self.make()

    def make(self):
        """
            Createa a grid of parameter values
        """
        GRID = []  #  list of params dictionaries
        Q = np.diag([1, 1, 20, 1, 1, 0, 0]) * 1e4
        count = 0

        for R_run in track(
            self.ranges["R_run"], description="generating grid"
        ):
            params = self._params.copy()
            params["CONTROL_CONFIG"]["R_run"] = np.diag([1, 1, 1]) * R_run

            for planning in self.ranges["planning"]:
                params["PLANNING_CONFIG"]["prediction_length_run"] = int(
                    planning
                )

                for Z_run in self.ranges["Z_run"]:
                    params["CONTROL_CONFIG"]["Z_run"] = (
                        np.diag([1, 1, 1]) * Z_run
                    )

                    for q0 in self.ranges["Q_0"]:
                        params["CONTROL_CONFIG"]["Q"] = Q.copy()
                        params["CONTROL_CONFIG"]["Q"][:, 0] = q0 * 1e4

                        for q1 in self.ranges["Q_1"]:
                            params["CONTROL_CONFIG"]["Q"][:, 1] = q1 * 1e4

                            for q3 in self.ranges["Q_3"]:
                                params["CONTROL_CONFIG"]["Q"][:, 3] = q3 * 1e4

                                for q4 in self.ranges["Q_4"]:
                                    params["CONTROL_CONFIG"]["Q"][:, 4] = (
                                        q4 * 1e4
                                    )
                                    GRID.append(params)

                                    # save
                                    self.save_params_and_bash(params, count)
                                    count += 1

        logger.debug(
            f"Generated GRID with {count} parameters combinations [{count * 5} SIMULATIONS]"
        )
        runtime_min = count * 5 * 20
        runtim_h = str(timedelta(minutes=runtime_min))
        runtim_h_parallel = str(
            timedelta(minutes=runtime_min / 100)
        )  # assume 100 sims run at once
        logger.debug(
            f'Expected total runtime: "{runtim_h}" | "{runtim_h_parallel}" with 100 sims in parallel'
        )

    def save_params_and_bash(self, params, count):
        # save JSON
        config_path = (
            self.save_path / "config_files" / f"control_GRID_{count}.json"
        )
        to_json(clean(params), config_path)

        # save BASH
        hpc_config_path = (
            paths.winstor_main.parent
            / "control_grid_search"
            / "config_files"
            / config_path.name
        )
        bash_path = self.save_path / "bash_files" / f"control_GRID_{count}.sh"

        bash = BASH.replace("CONFIG_PATH", f'"{hpc_config_path}"').replace(
            "\\", "/"
        )

        with open(bash_path, "w") as out:
            out.write(bash)


if __name__ == "__main__":
    save_path = Path(
        "Z:\\swc\\branco\\Federico\\Locomotion\\control\\control_grid_search"
    )

    Grid(save_path=save_path)
