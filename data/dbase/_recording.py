import pandas as pd
import sys
from pathlib import Path
from typing import Union
from loguru import logger
import h5py
import numpy as np
from rich.prompt import Confirm

sys.path.append("./")

from fcutils.path import size
from fcutils.maths.signals import get_onset_offset
from data.dbase.quality_control import load_or_open
from data.dbase.io import get_recording_local_copy


def get_tscale(ephys_ap_data_path, ai_file_path, sampling_rate=30000):
    """
        Because of a mistake we don't have time scaling information in validated sessions, 
        so have to extract it anew
    """
    if "220220_517" in ephys_ap_data_path:
        ephys_ap_data_path = ephys_ap_data_path.replace(
            "220220_517", "220120_517"
        )

    ai_file_path = Path(ai_file_path)
    if not ai_file_path.exists():
        ai_file_path = Path(r"K:\analog_inputs_temp") / ai_file_path.name

    # load analog from bonsai
    try:
        bonsai_probe_sync = load_or_open(
            ephys_ap_data_path, "bonsai", ai_file_path, 3
        )
    except FileNotFoundError as e:
        # raise FileNotFoundError(e)
        logger.warning(f"Failed to find recording data: {e}")

    # load data from ephys (from local file if possible)
    ephys_probe_sync = load_or_open(
        ephys_ap_data_path,
        "ephys",
        get_recording_local_copy(ephys_ap_data_path),
        -1,
        order="F",
        dtype="int16",
        nsigs=385,
    )
    if ephys_probe_sync is None:
        logger.warning(
            f"Could not open ephys probe sync file: {ephys_ap_data_path}"
        )  #
        return None, None

    # get sync pulses for bonsai and ephys
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(
        bonsai_probe_sync, 4
    )

    ephys_sync_onsets, ephys_sync_offsets = get_onset_offset(
        ephys_probe_sync, 45
    )

    # remove pulses that are too brief
    errors = np.where(np.diff(bonsai_sync_onsets) < sampling_rate / 3)[0]
    bonsai_sync_offsets = np.delete(bonsai_sync_offsets, errors)
    bonsai_sync_onsets = np.delete(bonsai_sync_onsets, errors)

    # remove pulses that are too brief
    errors = np.where(np.diff(ephys_sync_onsets) < sampling_rate / 2)[0]
    ephys_sync_offsets = np.delete(ephys_sync_offsets, errors)
    ephys_sync_onsets = np.delete(ephys_sync_onsets, errors)

    tscale = (bonsai_sync_offsets[-1] - bonsai_sync_onsets[0]) / (
        ephys_sync_offsets[-1] - ephys_sync_onsets[0]
    )
    print("done getting tscale")
    return tscale, ai_file_path


def get_recording_filepaths(
    key: dict,
    rec_metadata: pd.DataFrame,
    recordings_folder: Path,
    rec_folder: str,
) -> dict:

    # check if recording has been validated
    metadata = rec_metadata.loc[
        rec_metadata["recording folder"] == rec_folder
    ].iloc[0]

    if metadata.Validated != "yes" or metadata["USE?"] != "yes":
        logger.debug(
            f'Recording for {key["name"]} was not validated - skipping.'
        )
        return

    if metadata["spike sorted"] != "yes":
        logger.debug(
            f'Recording for {key["name"]} not yet spike sorted - skipping.'
        )
        return

    rec_name = rec_metadata.loc[
        rec_metadata["recording folder"] == rec_folder
    ]["recording folder"].iloc[0]

    rec_path = recordings_folder / Path(rec_name) / Path(rec_name + "_imec0")
    key["concatenated"] = -1

    # complete the paths to all relevant files
    key["spike_sorting_params_file_path"] = str(
        rec_path / (rec_name + "_t0.imec0.ap.prm")
    )
    key["spike_sorting_spikes_file_path"] = str(
        rec_path / (rec_name + "_t0.imec0.ap.csv")
    )
    key["spike_sorting_clusters_file_path"] = str(
        rec_path / (rec_name + "_t0.imec0.ap_res.mat")
    )

    if key["name"] == "FC_220120_BAA110517_hairpin":
        key = {
            k: v.replace("220220", "220120") if isinstance(v, str) else v
            for k, v in key.items()
        }

    for name in (
        "spike_sorting_params_file_path",
        "spike_sorting_spikes_file_path",
        "spike_sorting_clusters_file_path",
    ):
        if not Path(key[name]).exists():
            logger.warning(
                f'Cant find file for "{name}" in session "{key["name"]}" - maybe not spike sorted yet?\nPath: {key[name]}'
            )
            if Confirm.ask("Insert placeholder?"):
                pass
            else:
                return None

    # get probe configuration
    key["recording_probe_configuration"], key["reference"] = metadata[
        "probe config"
    ].split("_")
    return key


def load_cluster_curation_results(
    results_filepath: Union[Path, str], results_csv_path: Union[Path, str]
) -> list:
    # get clusters annotations
    """
        To get the class of each cluster (single vs noise), loop over
        the clusterNotes entry of the results file and for each cluster
        decode the h5py reference as int -> bytes -> str to see 
        what the annotation was
    """
    clusters_annotations = {}

    if not Path(results_filepath).exists():
        results_filepath = str(
            Path(r"M:\recordings_temp") / Path(results_filepath).name
        )

        results_csv_path = str(
            Path(r"M:\recordings_temp") / Path(results_csv_path).name
        )

    with h5py.File(results_filepath, "r") as mat:
        for clst in range(len(mat["clusterNotes"])):
            vals = mat[mat["clusterNotes"][()][clst][0]][()]
            if len(vals) < 3:
                continue
            else:
                note = "".join([bytes(v).decode("utf-8")[0] for v in vals])
                if note == "single":
                    clusters_annotations[clst + 1] = note

        logger.debug(
            f'Opened clustering results file, found {len(mat["clusterNotes"])} annotations'
        )
    logger.debug(f"Found {len(clusters_annotations)} single units")

    # load spikes data from the .csv file
    logger.debug(f"Loading spikes data from CSV ({size(results_csv_path)})")
    spikes = pd.read_csv(results_csv_path)
    logger.debug(
        f"     loaded data about {len(spikes)} spikes ({spikes.iloc[-1].spikeTimes}s)"
    )

    # get units data
    units = []
    for cluster in clusters_annotations.keys():
        unit = spikes.loc[spikes.spikeClusters == cluster]
        units.append(
            dict(
                raw_spikes_s=unit["spikeTimes"].values,
                unit_id=cluster,
                recording_site_id=unit.spikeSites.mode().iloc[0],
                secondary_sites_ids=np.sort(unit.spikeSites.unique()),
            )
        )

    return units


def get_unit_spike_times(
    unit: dict, triggers: dict, sampling_rate: int, tscale: float,
) -> dict:
    """
        Gets a unit's spikes times aligned to bonsai's video frames
        in both milliseconds from the recording start and in frame numbers
    """

    # take spike times in seconds, convert to samples number (in ephys samples)
    spikes_samples_ephys = unit["raw_spikes_s"] * sampling_rate

    # cut spikes in the 1s before/after recording
    # spikes_samples_ephys = spikes_samples_ephys[(spikes_samples_ephys > sampling_rate) & (spikes_samples_ephys < (spikes_samples_ephys[-1]-sampling_rate))]
    spikes_samples_ephys = spikes_samples_ephys[
        (spikes_samples_ephys > triggers["ephys_cut_start"])
        & (spikes_samples_ephys < triggers["ephys_cut_end"])
    ]

    # get number of samples relative to the first trigger
    spikes_samples_ephys -= triggers["ephys_cut_start"]

    # convert to samples numbers in bonsai samples
    spikes_samples = spikes_samples_ephys * tscale

    # we have spikes in samples relative to first probe syn trigger, now adjust relative to start of experiment
    spikes_samples += triggers["frame_to_drop_pre"] * 1 / 60 * sampling_rate

    # get the closest frame trigger to each spike sample
    samples_per_frame = 1 / 60 * sampling_rate
    spikes_frames = np.round(spikes_samples / samples_per_frame).astype(
        np.int64
    )

    # return data
    return dict(
        spikes_ms=spikes_samples / sampling_rate * 1000, spikes=spikes_frames
    )


# -------------------------------- firing rate ------------------------------- #


def gaussian(x, s):
    return (
        1.0
        / np.sqrt(2.0 * np.pi * s ** 2)
        * np.exp(-(x ** 2) / (2.0 * s ** 2))
    )


def calc_firing_rate(spikes_train: np.ndarray, dt: int = 10):
    """
        Computes the firing rate given a spikes train (wether there is a spike or not at each ms).
        Using a gaussian kernel with standard deviation = dt/2 [dt is in ms]
    """
    # create kernel & get area under the curve
    k = np.array(
        [gaussian(x, dt / 2) for x in np.linspace(-2 * dt, 2 * dt, dt)]
    )
    auc = np.trapz(k)

    # get firing rate
    frate = (
        np.convolve(spikes_train, k, mode="same") / auc * 1000
    )  # times 1000 to go from ms to seconds
    return frate


def get_units_firing_rate(
    info: dict, unit: dict, frate_window: float,
) -> np.ndarray:
    """
        Computs the firing rate of a unit by binnig spikes with bins
        of width = frate_window milliseconds. It also samples the resulting firing rate array
        to the firing rate at frame times
    """
    n_ms = unit["duration"] * 1000
    spikes_ms = unit["spikes_ms"].astype(np.int64)
    spikes = np.zeros(np.max([n_ms, spikes_ms[-1] + 1]))
    spikes[spikes_ms] = 1

    # get frate at each millisecond
    frate = calc_firing_rate(spikes, dt=frate_window)

    # get frate at each frame
    ms_per_frame = 1000 / 60
    idxs = np.round(np.arange(0, n_ms, step=ms_per_frame)).astype(np.int64)
    frate_frames = frate[idxs]
    if len(frate_frames) < unit["n_frames"]:
        frate_frames = np.pad(
            frate_frames, (0, unit["n_frames"] - len(frate_frames)), "constant"
        )
    return frate_frames


if __name__ == "__main__":
    pth = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap_res.mat"
    )
    excel_path = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap.csv"
    )
    load_cluster_curation_results(pth, excel_path)
