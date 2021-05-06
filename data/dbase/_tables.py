from loguru import logger
import shutil
import numpy as np

from fcutils.path import files, size

from data.paths import raw_data_folder


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
    logger.info("Sorting raw files")
    fls = files(raw_data_folder / "tosort")

    raise NotImplementedError("Bonsai should save .csv files too now")

    if isinstance(fls, list):
        for f in fls:
            src = raw_data_folder / "tosort" / f.name

            if f.suffix == ".avi":
                dst = raw_data_folder / "video" / f.name
            elif f.suffix == ".bin" or f.suffix == ".csv":
                dst = raw_data_folder / "analog_inputs" / f.name
            else:
                logger.info(f"File not recognized: {f}")
                continue

            logger.info(f"Moving file {src} to {dst}")
        shutil.move(src, dst)


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
    except:
        if dataname in list(table.fetch(checktag)):
            logger.debug("Entry with id: {} already in table".format(dataname))
        else:
            logger.debug(table)
            raise ValueError(
                "Failed to add data entry {}-{} to {} table".format(
                    checktag, dataname, table.full_table_name
                )
            )
