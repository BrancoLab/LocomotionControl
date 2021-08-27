import pandas as pd
import sys
from pathlib import Path
from typing import Union
from loguru import logger
import h5py
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

sys.path.append('./')

def load_cluster_curation_results(results_filepath: Union[Path, str], results_csv_path: Union[Path, str]) -> list:
    # get clusters annotations
    '''
        To get the class of each cluster (single vs noise), loop over
        the clusterNotes entry of the results file and for each cluster
        decode the h5py reference as int -> bytes -> str to see 
        what the annotation was
    '''
    clusters_annotations = {}
    with h5py.File(results_filepath, 'r') as mat:
        for clst in range(len(mat['clusterNotes'])):
            vals = mat[mat['clusterNotes'][()][clst][0]][()]
            if len(vals) < 3: 
                continue
            else:
                note = ''.join([bytes(v).decode('utf-8')[0] for v in vals])
                if note == 'single':
                    clusters_annotations[clst + 1] = note

        logger.debug(f'Opened clustering results file, found {len(mat["clusterNotes"])} annotations')
    logger.debug(f'Found {len(clusters_annotations)} single units')

    # load spikes data from the .csv file
    logger.debug('Loading spikes data from CSV')
    spikes = pd.read_csv(results_csv_path)

    # get units data
    units = []
    for cluster in clusters_annotations.keys():
        unit = spikes.loc[spikes.spikeClusters == cluster]
        units.append(dict(
            raw_spikes_s=unit['spikeTimes'].values,
            unit_id = cluster,
            recording_site_id = unit.spikeSites.mode().iloc[0],
        ))
        
    return units


def get_unit_spike_times(unit:dict, triggers:dict, sampling_rate:int)->dict:
    '''
        Gets a unit's spikes times aligned to bonsai's video frames
        in both milliseconds from the recording start and in frame numbers
    '''
    logger.debug(f'         getting spikes times')
    # get spike times in sample number
    spikes_samples = unit['raw_spikes_s'] * sampling_rate

    # cut and scale to match bonsai data
    spikes_samples = spikes_samples - triggers['ephys_cut_start']
    spikes_samples = spikes_samples[spikes_samples>0]
    spikes_samples = spikes_samples * triggers['ephys_time_scaling_factor']

    # get the closest frame trigger to each spike sample
    samples_per_frame = 1 / 60 * sampling_rate
    spikes_frames = np.floor(spikes_samples / samples_per_frame).astype(np.int64)

    if np.any(spikes_frames < 0) or np.any(spikes_frames > triggers['trigger_times'].max()):
        raise ValueError('Error while assigning frame number to spike times')

    # return data
    return dict(spikes_ms  = unit['raw_spikes_s'] * 1000, spikes=spikes_frames)

def get_unit_firing_rate(unit_spikes:dict, triggers:dict, filter_width:int, sampling_rate:int) -> dict:
    '''
        Given the times at which a unit spikes, get the firing rate.
    '''
    logger.debug(f'         getting spikerate')
    # get raster
    last_spike = int(unit_spikes['spikes_ms'][-1])
    time_array = np.zeros(last_spike + 1000)
    time_array[unit_spikes['spikes_ms'].astype(np.int64)] = 1

    # get spike rate
    spike_rate = gaussian_filter1d(time_array, filter_width)

    # get spike rate at frame times
    video_triggers = (triggers['trigger_times']/sampling_rate).astype(np.int64)
    spike_rate_frames = spike_rate[video_triggers]

    return dict(spikerate_ms=spike_rate[:-1000], spikerate=spike_rate_frames)


if __name__ == '__main__':
    pth = Path(r'W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap_res.mat')
    excel_path = Path(r'W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap.csv')
    load_cluster_curation_results(pth, excel_path)