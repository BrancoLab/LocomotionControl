import pandas as pd
import sys
from pathlib import Path
from typing import Union
from loguru import logger
import h5py

sys.path.append('./')

def load_cluster_curation_results(results_filepath: Union[Path, str], results_csv_path: Union[Path, str]):
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
            vals = list(mat[mat['clusterNotes'][()][clst][0]])
            if len(vals) < 3: 
                clusters_annotations[clst] = 'none'
                continue
            else:
                clusters_annotations[clst] = ''.join(bytes(v).decode('utf-8') for v in vals)

        logger.info(f'Opened clustering results file, found {len(mat["clusterNotes"])} annotations and {len(mat["clusterSites"])} cluster sites data')

    # TODO get the recording site of each good cluster
    # TODO load data from .csv file
    # TODO organize spikes by good clusters
    # TODO return stuff and save

    # load spikes data from the .csv file
    spikes = pd.read_csv(results_csv_path)

    a = 1




if __name__ == '__main__':
    pth = Path(r'W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap_res.mat')
    excel_path = Path(r'W:\swc\branco\Federico\Locomotion\raw\recordings\210713_750_longcol_intref_openarena_g0\210713_750_longcol_intref_openarena_g0_imec0\210713_750_longcol_intref_openarena_g0_t0.imec0.ap.csv')
    load_cluster_curation_results(pth, excel_path)