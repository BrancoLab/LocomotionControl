using DataFrames, Plots

# ---------------------------------------------------------------------------- #
#                                    PYCALL                                    #
# ---------------------------------------------------------------------------- #

using PyCall

@pyinclude("../data/dbase/djconn.py")

py"""
import sys
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
from data.dbase.db_tables import (
    LocomotionBouts,
    ProcessedLocomotionBouts,
    SessionCondition,
)
"""

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #


roi_limits = Dict(  # extent of each curve's ROI in global coords
    1 => (0, 12),   # in global coordinates
    2 => (12, 24),
    3 => (24, 36),
    4 => (36, 48),
)



# --------------------------------- plotting --------------------------------- #
roi_colors = [:red, :green, :blue, :orange]


arena_ax_kwargs = Dict(
    :xlim=>[-5, 45], :ylim=>[-5, 65], :grid=>false, :aspect_ratio=>:equal, 
    :xlabel=>"x [cm]", :ylabel=>"y [cm]"
)



include("utils.jl")