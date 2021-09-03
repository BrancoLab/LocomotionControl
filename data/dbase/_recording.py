import pandas as pd
import sys
from pathlib import Path
from typing import Union, Tuple
from loguru import logger
import h5py
import numpy as np

sys.path.append("./")

from fcutils.maths.signals import rolling_mean
from fcutils.path import files, size

def get_recording_filepaths(key:dict, rec_metadata:pd.DataFrame, recordings_folder:Path, rec_folder:str) -> dict:

    # Check if it's a concatenated recording
    concat_filepath = rec_metadata.loc[
        rec_metadata["recording folder"] == rec_folder
    ]["concatenated recording file"].iloc[0]
    if isinstance(concat_filepath, str):
        # it was concatenated
        rec_name = concat_filepath
        rec_path = recordings_folder / Path(rec_name)
        key["concatenated"] = 1
        
        if not rec_path.is_dir() or not files(rec_path):
            logger.warning(f'Invalid rec path: {rec_path} for session {key["name"]}')
            return None
            
        rec_name = rec_name + '_g0'
    else:
        rec_name = rec_metadata.loc[
            rec_metadata["recording folder"] == rec_folder
        ]["recording folder"].iloc[0]
        rec_path = (
            recordings_folder
            / Path(rec_name)
            / Path(rec_name + "_imec0")
        )
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

    for name in (
        "spike_sorting_params_file_path",
        "spike_sorting_spikes_file_path",
        "spike_sorting_clusters_file_path",
    ):
        if not Path(key[name]).exists():
            logger.warning(f'Cant file for "{name}" in session {key["name"]}')
            return None

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
    logger.debug(f"     loaded data about {len(spikes)} spikes ({spikes.iloc[-1].spikeTimes}s)")

    # get units data
    units = []
    for cluster in clusters_annotations.keys():
        unit = spikes.loc[spikes.spikeClusters == cluster]
        units.append(
            dict(
                raw_spikes_s=unit["spikeTimes"].values,
                unit_id=cluster,
                recording_site_id=unit.spikeSites.mode().iloc[0],
                secondary_sites_ids =np.sort(unit.spikeSites.unique())
            )
        )

    return units


def get_unit_spike_times(
    unit: dict, triggers: dict, sampling_rate: int, pre_cut:int=None, post_cut:int=None
) -> dict:
    """
        Gets a unit's spikes times aligned to bonsai's video frames
        in both milliseconds from the recording start and in frame numbers
    """
    logger.debug(f"         getting spikes times")

    # check that the last spike didn't occur too late
    # if unit["raw_spikes_s"][-1] - 2 > triggers['duration']:  # -2 to account for tails of recording
    #     raise ValueError('The last spike cannot be after after the end of the video')


    # get spike times in sample number
    spikes_samples = unit["raw_spikes_s"] * sampling_rate

    # cut to deal with concatenated recordings
    if pre_cut is not None:
        spikes_samples = spikes_samples[(spikes_samples > pre_cut) & (spikes_samples < post_cut)]
        spikes_samples -= pre_cut

    # cut frames spikes in the 1s before/after the recording
    spikes_samples = spikes_samples[(spikes_samples > triggers["ephys_cut_start"]) & (spikes_samples < triggers["n_samples"])]

    # cut and scale to match bonsai data
    spikes_samples = spikes_samples - triggers["ephys_cut_start"]
    spikes_samples *= triggers["ephys_time_scaling_factor"]

    # get the closest frame trigger to each spike sample
    samples_per_frame = 1 / 60 * sampling_rate
    spikes_frames = np.floor(spikes_samples / samples_per_frame).astype(
        np.int64
    )

    if (
        np.any(spikes_frames < 0)
        or np.any(spikes_frames > triggers["trigger_times"].max())
        or spikes_frames.max() > triggers["n_frames"]
    ):
        raise ValueError("Error while assigning frame number to spike times")

    # return data
    return dict(spikes_ms=spikes_samples / sampling_rate * 1000, spikes=spikes_frames)


def cut_concatenated_units(recording:dict, triggers:dict, rec_metadata:pd.DataFrame) -> Tuple[int, int]:
    '''
        Split units spiking data from concatenated recordings. Return min and max values (in samples)
        for spikes to keep. The keeping of the spikes is actually done in 'get_unit_spike_times'
    '''
    # get if first or second in concatenated data
    concat_filename = Path(recording['spike_sorting_spikes_file_path']).stem[:-15]
    concat_metadata = rec_metadata.loc[rec_metadata['concatenated recording file']==concat_filename]

    if len(concat_metadata) != 2:
        raise ValueError('Expected to find two metadata entries')

    rec_filename = Path(recording['ephys_ap_data_path']).stem[:-12]
    idx = np.where(concat_metadata['recording folder'] == rec_filename)[0][0]
    is_first = idx == 0

    # deal with things separately based on if its first or second recording
    if is_first:
        pre_cut = 0
        post_cut = triggers['n_samples']
    else:
        # Get the number of samples of the previous recording and set that as pre_cut
        raise NotImplementedError('need to deal with second of two concatenated recordings')

    return pre_cut, post_cut


# -------------------------------- firing rate ------------------------------- #


def get_units_firing_rate(units:Union[pd.DataFrame, dict], frate_window:float, triggers:dict, sampling_rate:int) -> Union[pd.DataFrame, dict]:
    '''
        Computs the firing rate of a unit by binnig spikes with bins
        of width = frate_window milliseconds. It also samples the resulting firing rate array
        to the firing rate at frame times
    '''
    # prepare arrays
    logger.debug(f'Getting firing rate with window width {frate_window}ms')
    trigger_times_ms = np.int32(triggers['trigger_times'] / sampling_rate * 1000)
    trigger_times_ms[-1] = trigger_times_ms[-1] - 1
    n_ms = int(triggers['trigger_times'][-1] / sampling_rate * 1000)
    time_array = np.zeros(n_ms)

    # check if we are dealing with a single unit
    if isinstance(units, dict):
        units_list = [pd.Series(units)]
    else:
        units_list = [unit for i,unit in units.iterrows()]

    # get the expected number of bins
    n_bins = int(np.ceil(n_ms/frate_window))

    # iterate over units
    rates, rates_frames = [], []
    for i, unit in enumerate(units_list):
        spikes_ms = unit.spikes_ms.astype(np.int32)
        if spikes_ms.max() > n_ms: 
            raise ValueError('spikes times after max duration')

        # get an array with number of spikes per ms
        spikes_ms_counts = pd.Series(spikes_ms).value_counts()
        spikes_counts = time_array.copy()
        spikes_counts[spikes_ms_counts.index] = spikes_ms_counts.values

        # convolve with gaussian
        spike_rate = rolling_mean(spikes_counts, frate_window)
        if not len(spike_rate) == n_ms:
            raise ValueError('Should be of length n milliseconds')

        # get spike rate at frame times
        spike_rate_frames = spike_rate[trigger_times_ms]
        if len(spike_rate_frames) != len(triggers['trigger_times']):
            raise ValueError('Mismatched array length')

        rates.append(spike_rate)
        rates_frames.append(spike_rate_frames)

    units['firing_rate'] = rates_frames
    units['firing_rate_ms'] = rates

    return units

    





if __name__ == "__main__":
    pth = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap_res.mat"
    )
    excel_path = Path(
        r"W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap.csv"
    )
    load_cluster_curation_results(pth, excel_path)
