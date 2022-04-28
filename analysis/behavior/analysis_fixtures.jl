
cd("/Users/federicoclaudi/Documents/Github/LocomotionControl/analysis/behavior")
import Pkg; 
Pkg.activate(".")

using Plots, Statistics, Colors
import KernelDensity: kde
using MultiKDE

using Term
import Term: install_term_logger
install_term_logger()

import MyterialColors: salmon, green_dark, grey, grey_dark, grey_darker, black

import jcontrol.comparisons: ComparisonPoints, ComparisonPoint
import jcontrol: FULLTRACK, Solution
import jcontrol.io: load_cached_trials

"""
Useful things for behavior analysis. Mostly Used in Thesis/Chpt3 analyses.
"""

# ---------------------------------------------------------------------------- #
#                                    TRIALS                                    #
# ---------------------------------------------------------------------------- #
trials = load_cached_trials(; keep_n = nothing);
@info "Loaded $(length(trials)) trials"


S = getfield.(trials, :s)
X = getfield.(trials, :x)
Y = getfield.(trials, :y)
U = getfield.(trials, :u)
Ω = getfield.(trials, :ω)

# ---------------------------------------------------------------------------- #
#                               COMPARISON POINTS                              #
# ---------------------------------------------------------------------------- #
# load comparison points
cpoints = ComparisonPoints(FULLTRACK; δs=2, trials=trials, s₀=1.0, s₁=250.0);
cpoints_valid = cpoints.points[2:end-2]


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
        name="first", s0 = 15.0, s=35.0, sf=38.0, tpos=20.0, maxdur=1.5, direction=-1
    ),
    Curve(;
        name="second", s0 = 45.0, s=86.0, sf=96.0, tpos=55.0, maxdur=1.5, direction=-1
    ),
    Curve(;
        name="third", s0 = 98.0, s=137.0, sf=142.0, tpos=110.0, maxdur=1.5, direction=1
    ),
    Curve(;
        name="fourth", s0 = 150.0, s=188.0, sf=220.0, tpos=155.0, maxdur=1.5, direction=1
    ),
]




# ---------------------------------------------------------------------------- #
#                               GLOBAL SOLUTIONS                               #
# ---------------------------------------------------------------------------- #

function fix_solution_dtype(sol)
    # for col in names(sol)
    #     sol[:, col] = map(v-> v isa Number ? v : parse(Float64, v), sol[:, col])
    # end
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
    globalsolution = DataFrame(CSV.File(joinpath(PATHS["horizons_sims_cache"], "global_solution.csv")))
    return Solution(fix_solution_dtype(globalsolution));
end

function load_horizons_solutions()
    # load individual solutions
    files = sort(glob("multiple_horizons_mtm_horizon_length*.csv", PATHS["horizons_sims_cache"]), lt=natural)

    solutions_df = map(file->DataFrame(CSV.File(file)), files)
    solutions_df = map(sol -> fix_solution_dtype(sol), solutions_df)

    solutions = df2sol.(solutions_df)
    _names = map(file -> split(file, "_")[end][1:end-4], files);
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