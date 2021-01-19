import numpy as np
from loguru import logger
import sys
from pathlib import Path

from control import config
from control.manager import Manager


_log = Path("grid_search_log.log")
if _log.exists():
    _log.unlink()


logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add(str(_log), level="DEBUG")

"""
    Do a grid search of values of of R and W cost weights matrices
    to find good controls.
"""
marks = (0.0001, 0.001, 0.01, 0.1, 0.5, 1)
for R in marks:
    for W in marks:
        for itern in range(3):
            # set new values
            logger.debug(f"Values, R:{R}, W:{W} iteration:{itern}")
            config.CONTROL_CONFIG["R"] = np.diag([1, 1, 1]) * R
            config.CONTROL_CONFIG["W"] = np.diag([1, 1, 1]) * W
            config.MANAGER_CONFIG[
                "exp_name"
            ] = f'grid_search_R_{str(R).replace(".","_")}_W_{str(W).replace(".","_")}_iter_{itern}'
            logger.debug(config.MANAGER_CONFIG["exp_name"])

            # run
            try:
                man = Manager(winstor=True)
                man.run(n_secs=10)
            except Exception as e:
                logger.warning(
                    f"Simulation crashed at {man.itern} steps with exception: {e}"
                )

                # try wrapping up anyways
                try:
                    man.wrap_up()
                except Exception as e:
                    logger.warning(
                        f"Wrapping up simulation failed with exception: {e}"
                    )
