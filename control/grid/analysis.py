# %%
# import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from rich.progress import track
from rich.logging import RichHandler
from myterial import orange

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


from pyinspect.utils import dir_files, subdirs

"""
    Preprocess and analyze the results of the grid search for control parameters
"""
# %%

# ---------------------------------------------------------------------------- #
#                                 PREPROCESSING                                #
# ---------------------------------------------------------------------------- #
# get some paths
main_folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\control_grid_search"
)

simulations_fld = main_folder / "simulations"
analysis_fld = main_folder / "analysis"
analysis_fld.mkdir(exist_ok=True)

# %%
# Get all simulations directories
simulations_folders = subdirs(simulations_fld)
logger.info(f"Found {len(simulations_folders)} simulations folders")

# %%
# check all simulations are complete
good = True
for fld in track(simulations_folders, description="checking..."):
    subs = subdirs(fld)
    if len(subs) < 5:
        logger.info(f"[{orange}] Not all simulations ran for: {fld.name}")
        good = False

    for subfld in subs:
        if not len(dir_files(subfld)) > 3:
            logger.info(
                f"[{orange}] Incomplete simulation: {fld.name} - {subfld.name}"
            )
            good = False

if not good:
    logger.warning("[b red]At least one simulation is incomplete")
else:
    logger.info("[b green]All simulations are completed, yay!")

# %%
