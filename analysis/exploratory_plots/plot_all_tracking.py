import sys

sys.path.append("./")
import matplotlib.pyplot as plt
from pathlib import Path
from rich.progress import track

from data.dbase import db_tables

fld = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\all_tracking")


sessions = db_tables.Tracking().fetch("name")
print(f"Found {len(sessions)} sessions")

for session in track(sessions):
    savepath = fld / f"{session}.png"
    if savepath.exists():
        continue

    data = (
        db_tables.Tracking() * db_tables.Tracking.BodyPart()
        & "bpname='body'"
        & f"name='{session}'"
    )
    x = data.fetch("x")[0]
    y = data.fetch("y")[0]

    f, ax = plt.subplots(figsize=(8, 12))
    ax.plot(x, y, color="k", alpha=0.5)
    ax.set(title=session)

    #  save figure in fld
    f.savefig(savepath)
    plt.close(f)
