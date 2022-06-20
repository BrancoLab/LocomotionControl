"""
Save data from GLM folder .h5 files
to .parquet files in the population folder

"""
from pathlib import Path
import pandas as pd
import sys

cache = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys\GLM\data"
)
dest = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys\population\data"
)

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")


from analysis.ephys.utils import get_recording_names

for rec in get_recording_names():
    df = pd.read_hdf(cache / (rec + "_bouts.h5")).reset_index()

    df = df.groupby(df.index // 5).mean()  # average every 25ms, data at 200fps

    # renames columns to strings
    df.columns = df.columns.astype(str)
    df.to_parquet(dest / (rec + ".parquet"))
    del df
