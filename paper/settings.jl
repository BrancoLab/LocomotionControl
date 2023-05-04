using Pkg
Pkg.activate("../")

using MultivariateStats
using ManifoldLearning
using DataFrames, Plots, Statistics, KernelDensity, StatsPlots, CSV
import StatsBase: fit, ZScoreTransform, transform
import MyterialColors: Palette
using MyterialColors
using Term
install_term_logger()
install_term_stacktrace()
install_term_repr()

savefig_fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/GDL/plts"
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
    Surgery, 
    Tracking,
    TrackingBP,
    FiringRate,
    Unit,
    Probe,
)
"""

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #


roi_limits = Dict(  # extent of each curve's ROI in global coords
    1 => (0.05, 0.18, 0.14),   # in global coordinates s ∈ [0, 1] 
    2 => (0.20, 0.39, 0.34),   # start, stop, center
    3 => (0.41, 0.58, 0.54),
    4 => (0.58, 0.75, 0.73),
)




# --------------------------------- plotting --------------------------------- #
function palette_from_colors(c1::String, c2::String, n::Int)::Vector{String}
    palette = Palette(c1, c2; N=n)
    return getfield.(palette.colors, :string)
end

roi_colors  = palette_from_colors(indigo_light, indigo_darker, 4)

arena_ax_kwargs = Dict(
    :xlim=>[-5, 45], :ylim=>[-5, 65], :grid=>false, :aspect_ratio=>:equal, 
    :xlabel=>"x [cm]", :ylabel=>"y [cm]"
)

axes_fontsize= Dict(
    :xtickfontsize=>16,
    :ytickfontsize=>16,
    :ztickfontsize=>16,
    :xguidefontsize=>16,
    :yguidefontsize=>16,
    :zguidefontsize=>16,
    :legendfontsize=>16,
)

axes_kwargs = Dict(
    :right_margin => 12Plots.mm,
    :left_margin => 12Plots.mm,
    :top_margin => 12Plots.mm,
    :bottom_margin => 12Plots.mm,
    :grid => false,
)

variables_colors = Dict(
    :speed => :red,
    :acceleration => :blue,
    :angvel => :green,
    :angaccel => :purple,
)

variables_legends = Dict(
    :speed => "cm/s",
    :acceleration => "cm/s²",
    :angvel => "°/s",
    :angaccel => "°/s²",
)


include("utils.jl")
nothing