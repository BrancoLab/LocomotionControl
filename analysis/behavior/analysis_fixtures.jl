
cd("/Users/federicoclaudi/Documents/Github/LocomotionControl/analysis/behavior")
import Pkg; 
Pkg.activate(".")

using Plots, Statistics, Colors, StatsBase
import KernelDensity: kde
using MultiKDE
using DataFrames: DataFrame
using CSV
using Glob, NaturalSort

using Term
import Term: install_term_logger
install_term_logger()

import MyterialColors: salmon, green_dark, grey, grey_dark, grey_darker, black, blue_grey_darker, blue_dark

import jcontrol.comparisons: ComparisonPoints, ComparisonPoint
import jcontrol: FULLTRACK, Solution
import jcontrol.io: load_cached_trials
using jcontrol.visuals

"""
Useful things for behavior analysis. Mostly Used in Thesis/Chpt3 analyses.
"""
PLOTS_FOLDER = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/THESIS/Chpt3/Plots"


# ---------------------------------------------------------------------------- #
#                                    CURVES                                    #
# ---------------------------------------------------------------------------- #

"""
Stores information about a curve's position in the track
"""
struct Curve
    s0::Float64    # position of "start"
    s::Float64     # position of apex
    sf::Float64    # position of end
    maxdur::Float64
    name::String
    tpos::Float64   # where the angular velocity should be at 0 after previous turn
    direction::Int     # 1 for left turns, -1 for right turns

    cp_s0::ComparisonPoint
    cp_s::ComparisonPoint
    cp_sf::ComparisonPoint
end

function Curve(;
    s0::Float64,
    s::Float64 = 0.0,
    sf::Float64 = 0.0,
    maxdur::Float64 = 0.0,
    name::String = "",
    tpos::Float64 = 0.0,
    direction::Int = 0  ,
    )

    # get comparison points
    cpt = ComparisonPoints(FULLTRACK, [s0, s, sf])

    return Curve(s0, s, sf, maxdur, name, tpos, direction, cpt.points...)
end



curves = [
    Curve(;
        name="first", s0 = 0.0, s=32.0, sf=40.0, tpos=20.0, maxdur=1.5, direction=-1
    ),
    Curve(;
        name="second", s0 = 45.0, s=84.0, sf=96.0, tpos=60.0, maxdur=1.5, direction=-1
    ),
    Curve(;
        name="third", s0 = 98.0, s=135.0, sf=144.0, tpos=115.0, maxdur=1.5, direction=1
    ),
    Curve(;
        name="fourth", s0 = 150.0, s=187.0, sf=205.0, tpos=165.0, maxdur=1.5, direction=1
    ),
]

curves_colors  = range(HSL(360, .6, .7), stop=HSL(200, .5, .7), length=length(curves))

"""
Get the frames in a trial in which the mouse is between s0 and s
to get when in a `Curve`
"""
function get_curve_indices(trial, s0, s1)
    start = findlast(trial.s .< s0)
    start = isnothing(start) ? 1 : start
    start += 2

    stop = findfirst(trial.s[start:end] .>= s1) + start
    stop = isnothing(stop) ? length(trial.s) : stop
    stop -= 2

    idxs = repeat([false], length(trial.s))
    idxs[start:stop] .= true
    idx = findfirst(idxs)
    return idxs, idx, start, stop
end

# ---------------------------------------------------------------------------- #
#                                    TRIALS                                    #
# ---------------------------------------------------------------------------- #
all_trials = load_cached_trials(; keep_n = nothing);
@info "Loaded $(length(all_trials)) trials"




# ---------------------------------------------------------------------------- #
#                              TORTUOSITY ANALYSIS                             #
# ---------------------------------------------------------------------------- #    
function pathlength(x, y)
    return sum(
        sqrt.(diff(x).^2 .+ diff(y).^2)
    )
end

function linelength(x, y)
    return sqrt((x[end]-x[1])^2 + (y[end]-y[1])^2)
end


"""
Compute tortuosity along an XY, by taking the trace
in Δt intervals and computing the ratio of the path length 
vs the line length between the start and the end.

Assumes the trace is at 60fps
"""
function get_max_tortuosity(x, y; Δt=.25)
    T = 0:(1/60):(length(x)/60 - Δt)
    if length(T) == 0
        return 0
    end
    snapshots = T[1]:Δt:(T[end])

    tortuosity = []
    for tval in snapshots
        tf = tval + Δt
    
        i0 = findfirst(T .>= tval)
        i1 = max(length(x), findlast(T .<= tf))
    
        _x = x[i0:i1]
        _y = y[i0:i1]
        
        push!(tortuosity, pathlength(_x, _y) / linelength(_x, _y))
    end
    return max(tortuosity...)
end

# get the tortuosity at each curve for each trial
tortuosity = []
for (i, curve) in enumerate(curves)
    _tortuosity = []
    for trial in all_trials
        _, _, start, stop = get_curve_indices(trial, curve.s0, curve.sf)

        x = trial.x[start:stop]
        y = trial.y[start:stop]
    
        push!(_tortuosity, get_max_tortuosity(x, y; Δt=.1))
    end
    push!(tortuosity, _tortuosity)
end


# get which trials have low torosity
clean_trials = Dict{Any, Vector}(1=>[], 2=>[], 3=>[], 4=>[])
for (i, curve) in enumerate(curves)
    th = percentile(tortuosity[i], 97.5)
    for t in tortuosity[i]
        push!(clean_trials[i], t < th)
    end
    
end

# # mark trials that are clean at each curve
# clean_trials["wholetrial"] = .*(values(clean_trials)...)

# tot = length(all_trials)
# discarded = length(filter(i->clean_trials["wholetrial"][i] == 0, 1:tot))
# @info "After tortuosity analysis, discarded $(round(discarded/tot * 100; digits=3))% of trials | $(tot - discarded) trials left"

# filter trials
# trials = [t for (i,t) in enumerate(all_trials) if clean_trials["wholetrial"][i]]
trials = all_trials
# high_torosity_trials = [t for (i,t) in enumerate(all_trials) if clean_trials["wholetrial"][i] == 0]

# compute quantities on clean trials
S = getfield.(trials, :s)
X = getfield.(trials, :x)
Y = getfield.(trials, :y)
Θ = getfield.(trials, :θ)
U = getfield.(trials, :u)
Ω = getfield.(trials, :ω)

# ---------------------------------------------------------------------------- #
#                               COMPARISON POINTS                              #
# ---------------------------------------------------------------------------- #
function get_comparison_points(trials; δs=2.0)
    # load comparison points
    cpoints = ComparisonPoints(FULLTRACK; δs=δs, trials=trials, s₀=1.0, s₁=250.0);
    cpoints_valid = cpoints.points[2:end-2]
    return cpoints, cpoints_valid
end


# ---------------------------------------------------------------------------- #
#                                 MTM SOLUTIONS                                #
# ---------------------------------------------------------------------------- #
function fix_solution_dtype(sol)
    sol.Fu = map(v-> v isa Number ? v : parse(Float64, v), sol.Fu)
    sol
end

fixvec(vec) = map(v-> v isa Number ? v : parse(Float64, v), vec)

function df2sol(df)
    _keys = filter!(k -> Symbol(k) != :δt, names(df))
    dt = "δt" ∈ names(df) ? df[1, :δt] : 0.0

    return Solution(; Dict(map(k -> Symbol(k)=>fixvec(df[:, k]), _keys))..., δt=dt)
end


function load_global_solution()
    # gsol_path = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/horizons_mtm_sims_wholetrack/global_solution.csv"
    fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior"
    gsol_path = joinpath(fld, "globalsolution.csv")
    globalsolution = DataFrame(CSV.File(gsol_path))
    return Solution((df2sol ∘ fix_solution_dtype)(globalsolution));
end

function load_global_solution(gsol_path)
    globalsolution = DataFrame(CSV.File(gsol_path))
    return Solution((df2sol ∘ fix_solution_dtype)(globalsolution));
end

function load_mtm_solutions(; folder=nothing, name="multiple_horizons_mtm_horizon_length")
    # load individual solutions
    folder = isnothing(folder) ? PATHS["horizons_sims_cache"] : folder
    files = sort(glob("$(name)*.csv", folder), lt=natural)

    solutions_df = map(file->DataFrame(CSV.File(file)), files)
    solutions_df = map(sol -> fix_solution_dtype(sol), solutions_df)

    solutions = df2sol.(solutions_df)
    # _names = map(file -> split(file, "_")[end][1:end-4], files);
    _names = map(file -> basename(file), files)
    return solutions, _names
end


# ---------------------------------------------------------------------------- #
#                                   UTILITIES                                  #
# ---------------------------------------------------------------------------- #
calcspeed(solution, idx) = sqrt.(solution.u[idx] .^ 2 .+ solution.v[idx] .^ 2)

euclidean(x::Number, y::Number, x2, y2) = sqrt((x2-x)^2 + (y-y2)^2)
euclidean(x::Vector, y::Vector, x2, y2) = sqrt.((x .- x2) .^ 2 .+ (y .- y2) .^ 2)

filterdf(df, col, val) = df[df[:, col] .== val, :]
filterdf(df, col1, val1, col2, val2) = df[(df[:, col1] .== val1) .* (df[:, col2] .== val2), :]