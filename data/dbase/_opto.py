from loguru import logger
import pandas as pd

from data.dbase._tables import insert_entry_in_table


def fill_opto_table(table, session_table):
    # load metadata
    metadata = pd.read_excel(
        table.opto_session_metadata_file, engine="odf"
    )

    session_names = session_table().fetch("name")

    # fill in
    for i, row in metadata.iterrows():
        name = row["bonsai file name"].split("_video")[0]
        if name not in session_names:
            raise ValueError(f"Did not find session {name} in Session table, add it first.")
        
        key = dict(
            mouse = row["mouse"],
            name = name,
            laser_power = row["laser power (mW)"],
            roi_1 = row["ROI 1"],
            roi_2 = row["ROI 2"],
            roi_3 = row["ROI 3"],
            roi_4 = row["ROI 4"],
            roi_5 = row["ROI 5"],
        )

        insert_entry_in_table(key["name"], "name", key, table)

