from loguru import logger
import shutil
import numpy as np
from pathlib import Path
import pandas as pd

from fcutils.path import files, size
from fcutils.progress import track

from data.paths import raw_data_folder, local_raw_recordings_folder, probes_surgeries_metadata

def get_probe_metadata(mouse: str):
    metadata = pd.read_excel(
        probes_surgeries_metadata, engine="odf"
    )

    mouse_id = int(mouse[-3:])
    metadata = metadata.loc[metadata['Unnamed: 1'] == mouse_id]
    if metadata.empty:
        logger.debug(f'No probe implant metadata found for mouse: {mouse_id}')
        return None
    
    try:
        cleaned_data = dict(
            skull_coordinates = np.array([metadata['ADJUSTED coordinates'], metadata['Unnamed: 7']]),
            angle_ap = metadata['angles'].iloc[0],
            angle_ml = metadata['Unnamed: 10'].iloc[0],
            implanted_depth = metadata['inserted probe'].iloc[0]/1000,
            reconstructed_track_filepath = metadata['reconstructed probe file path'].iloc[0],
        )
    except TypeError:
        logger.debug(f'Incomplete probe implant metadata found for mouse: {mouse_id}')
        return None
    return cleaned_data

def get_recording_local_copy(remote_path):
    # trying to find a local copy of the file first
    remote = Path(remote_path)
    local_path = local_raw_recordings_folder / remote.parent / remote.name
    if local_path.exists():
        logger.debug(f"Using local copy of file: {local_path.name}")
        return local_path
    else:
        logger.warning(
            f"Could not find local copy of recording file: {local_path.name}"
        )
        return remote_path


def load_bin(filepath, nsigs=4, dtype=None, order=None):
    """
        loads and reshape a bonsai .bin file
    """
    logger.debug(f'Opening BIN file: "{filepath}" ({size(filepath)})')

    dtype = dtype or np.float64
    order = order or "C"

    with open(filepath, "r") as fin:
        data = np.memmap(fin, dtype=dtype, order=order, mode="r")

    return data.reshape(-1, nsigs)


def sort_files():
    """ sorts raw files into the correct folders """
    logger.info("Sorting raw behavior files")
    fls = files(raw_data_folder / "tosort")

    if isinstance(fls, list):
        logger.debug(f"Sorting {len(fls)} files")

        for f in track(fls, description="sorting", transient=True):
            src = raw_data_folder / "tosort" / f.name

            if f.suffix == ".avi":
                dst = raw_data_folder / "video" / f.name
            elif f.suffix == ".bin" or f.suffix == ".csv":
                dst = raw_data_folder / "analog_inputs" / f.name
            else:
                logger.warning(f"File not recognized: {f}")
                continue

            if dst.exists():
                logger.debug(f"Destinatoin file already exists, skipping")
            else:
                logger.info(f"Moving file '{src}' to '{dst}'")
                shutil.move(src, dst)
    else:
        logger.warning(f"Expected files list got: {fls}")


# For manual tables
def insert_entry_in_table(dataname, checktag, data, table, overwrite=False):
    """
        Tries to add an entry to a databse table taking into account entries already in the table

        dataname: value of indentifying key for entry in table
        checktag: name of the identifying key ['those before the --- in the table declaration']
        data: entry to be inserted into the table
        table: database table
    """
    if dataname in list(table.fetch(checktag)):
        return

    try:
        table.insert1(data)
        logger.debug("     ... inserted {} in table".format(dataname))
    except Exception as e:
        if dataname in list(table.fetch(checktag)):
            logger.debug("Entry with id: {} already in table".format(dataname))
        else:
            logger.debug(table)
            raise ValueError(
                "Failed to add data entry {}-{} to {} table with error\n{}".format(
                    checktag, dataname, table.full_table_name, e
                )
            )


def get_scorer_bodyparts(tracking):
    """
        Given the tracking data hierarchical df from DLC, return
        the scorer and bodyparts names
    """
    first_frame = tracking.iloc[0]
    try:
        bodyparts = first_frame.index.levels[1]
        scorer = first_frame.index.levels[0]
    except:
        raise NotImplementedError(
            "Make this return something helpful when not DLC df"
        )

    return scorer, bodyparts


def load_dlc_tracking(tracking_file: str):
    """
        load and unstack tracking data from a DLC file
    """
    tracking = pd.read_hdf(tracking_file)

    bodyparts = tracking.iloc[0].index.levels[1]
    scorer = tracking.iloc[0].index.levels[0]

    tracking = tracking.unstack()

    trackings = {}
    for bp in bodyparts:
        trackings[bp] = {
            c: tracking.loc[scorer, bp, c].values
            for c in ["x", "y", "likelihood"]
        }
    return trackings
