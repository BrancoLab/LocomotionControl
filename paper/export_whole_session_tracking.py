import sys

sys.path.append("./")

from time import sleep

# import data.dbase.djconn
from data.dbase.db_tables import (
    Tracking,
    TrackingBP,
    SessionCondition,
    Surgery,
)
import pandas as pd

dest_fld = "/Users/federicoclaudi/Desktop/mysql-server-locomotion/whole_session_tracking"

# check if the folder exists
import os

if not os.path.exists(dest_fld):
    os.makedirs(dest_fld)

entries = (
    Tracking * TrackingBP * SessionCondition * Surgery
    & "bpname='body'"
    & "target='CUN/GRN'"
)


count = 0
for entry in entries:
    count += 1
    # if count < 60:
    #     continue
    name, target, condition = (
        entry["name"],
        entry["target"],
        entry["condition"],
    )
    target = target.replace("/", "-")

    dest_fl = os.path.join(dest_fld, f"{name}_{target}_{condition}.csv")
    if os.path.exists(dest_fl):
        print(f"Skipping {name} {target} {condition}")
        sleep(0.25)
        continue

    print(count, name)

    df = pd.DataFrame(
        dict(
            x=entry["x"],
            y=entry["y"],
            u=entry["u"],
            udot=entry["udot"],
            speed=entry["speed"],
            acceleration=entry["acceleration"],
            theta=entry["theta"],
            thetadot=entry["thetadot"],
            thetadotdot=entry["thetadotdot"],
        )
    )

    df.to_csv(dest_fl)
