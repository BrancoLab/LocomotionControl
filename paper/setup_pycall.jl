# using Pkg
# Pkg.add("PyCall")
# ENV["PYTHON"] = raw"/Users/federicoclaudi/miniconda3/envs/gld/bin/python"
# Pkg.build("PyCall")


using PyCall

@pyinclude("data/dbase/djconn.py")

py"""
import sys
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
from data.dbase.db_tables import (
    Session,
    Recording
)

"""

sessions = py"pd.DataFrame(Session())"
print(sessions)