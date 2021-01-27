# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from rich.progress import track
from rich.logging import RichHandler
from myterial import orange
import pandas as pd
import sys
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error as MSE


from fcutils.maths.utils import derivative
from pyinspect.utils import dir_files, subdirs

sys.path.append("./")

from control.history import load_results_from_folder


logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


DO = {
    "checks": False,  # check that all simulations were completed
    "preprocess": False,  # preprocess data
    "analysis": True,
}

"""
    Preprocess and analyze the results of the grid search for control parameters
"""


def get_trajectory_at_each_frame(history, trajectory):
    """
        Returns an array with the values of the trajectory waypoint
        that the model is at at each frame
    """
    return trajectory[history.trajectory_idx.values]


def plot_histograms(df, columns, axarr, color, bins=None):
    """
        Plots histograms to show the distribution of 
        values for columns of a dataframe
    """
    out_bins = []
    for n, (col, ax) in enumerate(zip(columns, axarr.flatten())):
        data = df[col].values
        if col == "MSE":
            data[data > 2000] = 2100
        elif col == "omega_MSE":
            data[data > 6] = 7
        elif col == "control_jerk":
            data[data > 2000] = 2200
        elif col == "max_control":
            data[data > 3000] = 3100

        if bins is None:
            _bins = 50
        else:
            _bins = bins[n]
        _, _out_bins, _ = ax.hist(data, bins=_bins)
        out_bins.append(_out_bins)

        ax.set(xlabel=col, ylabel="count")
    return out_bins


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


# check all simulations are complete
if DO["checks"]:
    # Get all simulations directories
    simulations_folders = subdirs(simulations_fld)
    logger.info(f"Found {len(simulations_folders)} simulations folders")

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


# --------------------------- summarize simulations -------------------------- #
"""
    For each parameters combination, collate the results of each simulation
    and save as a dataframe
"""
if DO["preprocess"]:
    simulations_folders = subdirs(simulations_fld)
    logger.info(f"Found {len(simulations_folders)} simulations folders")

    logger.info("Collating data")
    for fld in track(simulations_folders, description="checking..."):
        out_path = analysis_fld / f"{fld.name}_results.h5"
        # if out_path.exists():
        #     continue

        _results = dict(
            rep=[],
            distance_travelled_ratio=[],
            n_waypoints_visited=[],
            MSE=[],
            omega_MSE=[],
            max_control=[],
            simulation_length=[],
            trajectory_length=[],
            control_jerk=[],
        )

        subs = subdirs(fld)
        for nrep, subfld in enumerate(subs):
            _results["rep"].append(nrep)

            # load data
            history, info, trajectory, _ = load_results_from_folder(subfld)

            # get trajectory length
            _results["trajectory_length"].append(trajectory.shape[0])

            # fill in empty history
            if history.empty:
                _results["distance_travelled_ratio"].append(0)
                _results["n_waypoints_visited"].append(0)
                _results["MSE"].append(
                    MSE(trajectory, np.zeros_like(trajectory))
                )
                _results["max_control"].append(1e4)
                _results["simulation_length"].append(0)
                _results["omega_MSE"].append(
                    MSE(trajectory[:, 4], np.zeros_like(trajectory[:, 4]))
                )
                _results["control_jerk"].append(1e3)
                continue

            # get simulation length
            _results["simulation_length"].append(len(history))

            # get distance of end state from origin (XY)
            end_distance_sim = euclidean(
                trajectory[0, :2], history.iloc[-1][["x", "y"]]
            )
            end_distance_traj = euclidean(
                trajectory[0, :2], trajectory[-1, :2]
            )
            _results["distance_travelled_ratio"].append(
                end_distance_sim / end_distance_traj
            )

            # get fraction of way points visited
            _results["n_waypoints_visited"].append(
                len(history.trajectory_idx.unique()) / trajectory.shape[0]
            )

            # get trajectory MSE
            traj_at_each_frame = get_trajectory_at_each_frame(
                history, trajectory
            )
            history_traj = history[
                ["x", "y", "theta", "v", "omega", "tau_r", "tau_l"]
            ].values
            _results["MSE"].append(MSE(history_traj, traj_at_each_frame))
            _results["omega_MSE"].append(
                MSE(history_traj[:, 4], traj_at_each_frame[:, 4])
            )

            # get max_control
            controls = history[["P", "N_r", "N_l"]].values
            _results["max_control"].append(np.max(np.abs(controls)))

            # get control jerk
            _results["control_jerk"].append(
                np.max(np.abs(derivative(controls, 0)))
            )

        # save dataframe
        pd.DataFrame(_results).to_hdf(out_path, key="hdf")


# %%

# ---------------------------------------------------------------------------- #
#                                   ANALYSIS                                   #
# ---------------------------------------------------------------------------- #
"""
    Get the results for params combination into a single dataframe, and do some plotting
    to see which is best.
"""
if DO["analysis"]:
    # get summary dataframe
    analysis_df_path = analysis_fld / "RESULTS.h5"

    if not analysis_df_path.exists():
        # Get each individual datafrme and collate the means
        results = pd.DataFrame(
            dict(
                params_n=[],
                distance_travelled_ratio=[],
                n_waypoints_visited=[],
                MSE=[],
                omega_MSE=[],
                max_control=[],
                simulation_length=[],
                trajectory_length=[],
                control_jerk=[],
            )
        )

        _files = dir_files(analysis_fld, "*_results.h5")
        for df in track(_files, description="Extracting"):
            df_name = df.stem
            df_id = int(df_name.split("_results")[0].split("_")[-1])

            df = pd.read_hdf(df, key="hdf")
            df["params_n"] = df_id
            del df["rep"]

            results = results.append(df.mean(), ignore_index=True)

        results.to_hdf(analysis_df_path, key="hdf")
    else:
        results = pd.read_hdf(analysis_df_path, key="hdf")

    # plot histograms
    logger.info("Making histograms")
    f, axarr = plt.subplots(ncols=4, nrows=2, figsize=(20, 9))
    col_names = results.columns[1:]

    # plot all data
    bins = plot_histograms(results, col_names, axarr, "b")

    # plot selected data
    selected = results.loc[results.distance_travelled_ratio > 0.8]
    selected = selected.loc[selected.n_waypoints_visited > 0.8]
    # selected = selected.loc[selected.MSE < 500]
    selected = selected.loc[selected.omega_MSE < 1.5]
    selected = selected.loc[selected.max_control < 1000]

    logger.debug(
        f"Selected {len(selected)} params combinations out of {len(results)}"
    )

    plot_histograms(selected, col_names, axarr, "salmon", bins=bins)

    # get scores
    values = {k: [] for k in results.params_n.values}
    weights = np.array([10, 3, 5])

    for metric in ("MSE", "omega_MSE", "control_jerk"):
        order = np.argsort(results[metric].values)[::-1]
        for n, param_n in enumerate(results.params_n[order].values):
            values[param_n].append(n)
    results["score"] = [
        np.mean(points * weights) for points in values.values()
    ]

    print(results.sort_values("score", inplace=False, ascending=False).head())

    plt.show()
