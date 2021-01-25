import click
from pyinspect import install_traceback
from pathlib import Path
import shutil
import sys

sys.path.append("./")

from control.manager import Manager
from control.paths import winstor_main

install_traceback(
    keep_frames=5, all_locals=True, relevant_only=False,
)


@click.command()
@click.option("--config", default=None, help="Config.json file")
def main(config):
    """
        Runs 5 simulations, with 5 different trajectories, for a choice
        of parameters. Called by a .sh file generated by grid.make_grid.GRID
    """
    grid_folder = winstor_main.parent / "control_grid_search"

    # Create a folder to save the data in
    name = Path(config).stem
    base = grid_folder / "simulations" / name

    # remove pre-existing folders and make a new one
    if base.exists():
        shutil.rmtree(str(base))
    base.mkdir(exist_ok=True)

    # Get the trajectories files
    traj_fld = grid_folder / "trajectories"
    trajectories = list(traj_fld.glob("*.npy"))

    # Run simulations
    for rep, traj_file in enumerate(trajectories):
        folder = base / f"{name}_rep_{rep}"
        Manager(
            winstor=True,
            folder=folder,
            config_file=config,
            to_db=False,
            trajectory_file=traj_file,
        ).run(n_secs=12)


if __name__ == "__main__":
    main()
