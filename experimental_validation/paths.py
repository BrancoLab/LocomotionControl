import sys
from pathlib import Path

"""
    Paths used for the analysis of 2WDD and other experimental validation
    data
"""
if sys.platform == "win32":
    folder_2WDD = Path(
        "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD"
    )
    DB_folder = Path(
        "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\experimental_validation\\2WDD"
    )
else:
    raise NotImplementedError

RAW = folder_2WDD / "RAW"
TRACKING_DATA = folder_2WDD / "TRACKING_DATA"
TRIALS_CLIPS = folder_2WDD / "TRIALS_CLIPS"

trials_records = folder_2WDD / "trials_records.json"

analysis_folder = DB_folder / "analysis"
