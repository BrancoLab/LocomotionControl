module comparisons
"""
    Code to facilitate the comparison of models with data.
"""

using Interpolations
import DataFrames: DataFrame
import MyterialColors: salmon, blue, green, purple, teal, indigo_dark
import Statistics: mean, std, median


import jcontrol: Track, trim, interpolate_wrt_to, closest_point_idx, int, FULLTRACK

export ComparisonPoint, ComparisonPoints, TrackSegment, track_segments



# ---------------------------------------------------------------------------- #
#                               COMPARISON POINTS                              #
# ---------------------------------------------------------------------------- #

"""
Store parametersof normal distribution
"""
struct Dist
    name::Union{String, Symbol}
    μ
    med
    σ
    values::Vector{Float64}
end
Base.show(io::IO, dist::Dist) = print(io, "Dist: $(dist.name)(μ: $(round(dist.μ; digits=2)), σ: $(round(dist.σ; digits=2)))")
Dist(name, data::Vector) = Dist(name, mean(data), median(data), std(data), data)


"""
Get the number of standard deviations away from the mean
"""
σ(x::Number, dist::Dist; use=:μ) = (x - getfield(dist, use))/dist.σ

"""
Store values of kinematics variables from tracking data
"""
struct KinematicsValues
    x::Dist
    y::Dist
    θ::Dist
    ω::Dist
    u::Dist
    v::Dist
end

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

    kinematics::Union{KinematicsValues, Nothing}
end

function Base.show(io::IO, p::ComparisonPoint)
    return print(
        io,
        "ComparisonPoint s=$(round(p.s; digits=1)) @ xy=($(round(p.x; digits=1)), $(round(p.y; digits=1)))",
    )
end


function ComparisonPoint(s, x, y, θ, η, w; trials::Union{Nothing, Vector}=nothing)
    isnothing(trials) && return ComparisonPoint(s, x, y, θ, η, w, nothing)

    # get kinematic variables values at comparison point
    data = Dict(
        :x => [],
        :y => [],
        :θ => [],
        :ω => [],
        :u => [],
        :v => [],
    )

    for trial in trials
        idx = closest_point_idx(trial.x, x, trial.y, y)

        for var in keys(data)
            push!(data[var], eval(:($trial.$var[$idx])))
        end
    end    
    # Store mean and standard deviation
    kv = KinematicsValues(
        Dist(:x, data[:x]), Dist(:y, data[:y]), Dist(:θ, data[:θ]), Dist(:ω, data[:ω]), Dist(:u, data[:u]),  Dist(:v, data[:v])
    )
    return ComparisonPoint(s, x, y, θ, η, w, kv)
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
function ComparisonPoints(
    track::Track; δs=10.04, s₀=nothing, s₁=nothing, kwargs...
)::ComparisonPoints
    # get values of s for comparisons
    s₀ = isnothing(s₀) ? δs : (s₀-δs)
    s₁ = isnothing(s₁) ? track.S_f : s₁
    ŝ = s₀:δs:s₁

    ŝ = [s₀, ŝ..., s₁]

    return ComparisonPoints(track, collect(ŝ); kwargs...)
end

function ComparisonPoints(track::Track, svalues::Vector{Float64}; trials::Union{Nothing, Vector}=nothing)
    # create interpolation objects for each varianle
    x = interpolate_wrt_to(track.S, track.X)
    y = interpolate_wrt_to(track.S, track.Y)
    θ = interpolate_wrt_to(track.S, track.θ)
    η = interpolate_wrt_to(track.S, track.θ .+ π / 2)
    width = 2 # hardcoded

    # get points and return ComparisonPoints
    pts = map((_s) -> ComparisonPoint(_s, x(_s), y(_s), θ(_s), η(_s), width; trials=trials), svalues)

    return ComparisonPoints(svalues, pts)
end


# ---------------------------------------------------------------------------- #
#                                 TRACK SEGMENT                                #
# ---------------------------------------------------------------------------- #
struct TrackSegment
    track::Track  # stores a Track object with the segment track.
    s₀::Float64  # s-value of start
    s₁::Float64  # s-value of end
    checkpoints::Vector{ComparisonPoint}
    color::String
end

function TrackSegment(s₀::Float64, s₁::Float64, color::String; δs=5)
    # theres 262 waypoints in the track's npy file, the track's S_f is 259.
    # get track
    track = trim(FULLTRACK, s₀ * 259, (s₁ - s₀) * 259)
    # rack(; start_waypoint=wp₀, keep_n_waypoints=nwp)

    return TrackSegment(
        track, 
        s₀,
        s₁,
        ComparisonPoints(FULLTRACK; δs=δs, s₀=(s₀+.01)*FULLTRACK.S_f, s₁=(s₁-.01)*FULLTRACK.S_f).points,
        color
    )
end

track_segments = [
    TrackSegment(0.15, .30, salmon),
    TrackSegment(0.24, .43, blue),
    TrackSegment(0.40, .61, green),
    TrackSegment(0.55, .72, purple),
    TrackSegment(0.71, .86, indigo_dark),
    TrackSegment(0.82, .975, teal),
]

end
