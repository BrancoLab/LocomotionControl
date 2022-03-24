module comparisons
"""
    Code to facilitate the comparison of models with data.
"""

using Interpolations

import jcontrol: Track, interpolate_wrt_to

# ---------------------------------------------------------------------------- #
#                               COMPARISON POINTS                              #
# ---------------------------------------------------------------------------- #

"""
Represents a point at which the comparison is made.

s: s value of CP
x,y: position of corresponding track point
θ: angle of corrisponding track point + π/2
w: width, used for plotting.
"""
struct ComparisonPoint
    s::Float64
    x::Float64
    y::Float64
    θ::Float64
    η::Float64
    w::Float64
end

function Base.show(io::IO, p::ComparisonPoint)
    return print(
        io,
        "ComparisonPoint s=$(round(p.s; digits=1)) @ xy=($(round(p.x; digits=1)), $(round(p.y; digits=1)))",
    )
end

"""
Store a bunch of comparison points
"""
struct ComparisonPoints
    s::Vector{Float64}
    points::Vector{ComparisonPoint}
end

"""
Find values of s ∈ [0:S] at which to do comparisons.

If `mode=::equallyspaced` a bunch of s values are selected
along the entire track length, with spacing `δs`. 
Otherwise hand-defined values are used.
"""
function get_comparison_points(
    track::Track; δs=10.04, s₀=nothing, s₁=nothing
)::ComparisonPoints
    # get values of s for comparisons
    s₀ = isnothing(s₀) ? δs : s₀
    s₁ = isnothing(s₁) ? track.S_f : s₁
    ŝ = s₀:δs:s₁

    return get_comparison_points(track, ŝ)
end

function get_comparison_points(track::Track, svalues::Vector{Float64})
    # create interpolation objects for each varianle
    x = interpolate_wrt_to(track.S, track.X)
    y = interpolate_wrt_to(track.S, track.Y)
    θ = interpolate_wrt_to(track.S, track.θ)
    η = interpolate_wrt_to(track.S, track.θ .+ π / 2)
    width = 1.5 # hardcoded

    # get points and return ComparisonPoints
    pts = map((_s) -> ComparisonPoint(_s, x(_s), y(_s), θ(_s), η(_s), width), svalues)

    return ComparisonPoints(svalues, pts)
end

# ---------------------------------------------------------------------------- #
#                                 TRACK SEGMENT                                #
# ---------------------------------------------------------------------------- #
struct TrackSegment
    s₀::Float64  # s-value of start
    s₁::Float64  # s-value of end
    checkpoints::Vector{ComparisonPoint}
end

end
