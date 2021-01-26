import click
from pyinspect import install_traceback
from pathlib import Path
import shutil
import sys
from loguru import logger


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
    base.mkdir(exist_ok=True)

    # add logging
    logger.add(str(base / "log.log"), level="WARNING")
    logger.warning(f"unning simulations with config file: {config}")

    # remove pre-existing folders and make a new one
    # if base.exists():
    #     shutil.rmtree(str(base))
    if not base.exists():
        base.mkdir(exist_ok=True)

    # Get the trajectories files
    traj_fld = grid_folder / "trajectories"
    trajectories = list(traj_fld.glob("*.npy"))

    # Run simulations
    for rep, traj_file in enumerate(trajectories):
        folder = base / f"{name}_rep_{rep}"

        # check if simulation ran already
        if folder.exists():
            if (
                len([f for f in folder.glob("*") if f.is_file()]) > 3
            ):  # the simulation was already completed
                logger.warning(
                    f"Skipping rep {rep} because it was complete already"
                )
                continue
            else:  # simulation incomplete, remove folder
                logger.warning(
                    f"Rrep {rep} was not complete, removing previous run"
                )
                shutil.rmtree(str(folder))

        # if it didn't run the run it now
        logger.warning(f"Running simulation for rep: {rep}")
        Manager(
            winstor=True,
            folder=folder,
            config_file=config,
            to_db=False,
            trajectory_file=traj_file,
        ).run(n_secs=12)
        logger.warning(f"Completed simulation for rep: {rep}\n\n")


if __name__ == "__main__":
    main()
