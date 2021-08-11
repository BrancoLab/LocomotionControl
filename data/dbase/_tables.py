from loguru import logger
import pandas as pd

from tpd import recorder


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


def print_table_content_to_file(table, name):
    content = pd.DataFrame(table()).to_string()
    recorder.add_text(content, name)
