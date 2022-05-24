using NPZ
using Dierckx: Spline1D, derivative
using Interpolations



# -------------------------------- TRACK TYPE -------------------------------- #
struct Track
    X::Vector
    Y::Vector
    XY::Matrix
    curvature::Vector   # κ value at each waypoint
    N::Int
    P::Vector  # parameter values
    S_f::Float64
    S::Vector
    δs::Vector
    κ::Any
    width::Function
    θ::Vector
end

function Base.show(io::IO, track::Track)
    return print(
        io,
        "Track. \e[2m$(round(track.S_f)) cm long, $(track.N) waypoints.\e[0m",
    )
end

struct Border
    X::Vector
    Y::Vector
end

# ----------------------------------- utils ---------------------------------- #

"""
Get border lines by extruding along normal directoin
"""
function get_track_borders(track::Track)
    η = track.θ .+ π / 2
    dx = track.width.(track.S) .* cos.(η)
    dy = track.width.(track.S) .* sin.(η)

    left = Border(track.X .+ dx, track.Y .+ dy)
    right = Border(track.X .- dx, track.Y .- dy)

    return left, right
end

"""
Compute the curvature at each waypoint in the track
"""
function compute_curvature(t, x, y, tnew)
    xspl = Spline1D(t, x)
    yspl = Spline1D(t, y)
    dx = derivative(xspl, tnew; nu=1)
    dy = derivative(yspl, tnew; nu=1)
    ddx = derivative(xspl, tnew; nu=2)
    ddy = derivative(yspl, tnew; nu=2)
    return -((ddx .* dy) .- (ddy .* dx)) ./ (((dx) .^ 2 + (dy) .^ 2) .^ (3 / 2))
end

"""
  κ(s)
Curvature as a function of curvilinear distance along the track
"""
function κ(s, k, S_f)
    idx = (Int ∘ ceil)(s / S_f * length(k))
    idx = idx == 0 ? 1 : idx
    idx = idx >= length(k) ? idx - 1 : idx

    if idx > length(k)
        @warn "how" s k S_f idx
    end
    return k[idx]
end

"""
Value of the width factor of the track at various svalues
"""
width_values = [
    [0.0 1.2]
    [0.01 1.1]
    [0.05 1.00]
    [0.07 .9] # first narrow
    [0.11 1.00] # first narrow
    [0.15 1]  # end of frst narrow
    [0.20 1]
    [0.25 1.00]
    [0.30 1.1]   # middle of second curve
    [0.36 1.2]
    [0.45 1.1]  # second narrow
    [0.5 0.9]  # 
    [0.55 1.05]
    [0.6 1.0]  # end of second narrow
    [0.63 1.05]  
    [0.67 1.05]
    [0.7 1.1]
    [0.75 1.1]
    [0.80 1.1]  # second part of last curve
    [0.9 1.1]
    [0.92 1.1]
    [0.95 1.15]
    [0.98 1.2]
    [1 1.2]  # end
]


# ---------------------------------------------------------------------------- #
#                              TRACK CONSTRUCTORS                              #
# ---------------------------------------------------------------------------- #

"""
Construct `Track` out of a set of waypoints
"""
function Track(XY, s1::Float64; resolution=0.00001, const_width=false)
    X = movingaverage(XY[:, 1], 3)
    Y = movingaverage(XY[:, 2], 3)
    X, Y = upsample(X, Y; δp=resolution)


    # @assert length(X) == size(XY, 1)
    # get new points locations thorugh interpolation
    # X, Y = upsample(XY[:, 1], XY[:, 2]; δp=resolution)
    N = length(X)

    # get distance step between each track point + total length
    Δx, Δy = Δ(X), Δ(Y)
    δs = sqrt.(Δx .^ 2 .+ Δy .^ 2)
    S_f = sum(δs)

    S = cumsum(δs)  # curvilinear coordinates
    @assert S[1] == 0 && S[end] - S_f < 0.01 "ops. $(S[1]) $(S[end])==$(S_f)"

    # compute curvature
    P = range(0, 1; length=length(X))
    curvature = compute_curvature(P, X, Y, P)
    K(s) = κ(s, curvature, S_f)
    @assert length(curvature) == length(X)

    # compute orientation of each segment
    θ = unwrap(atan.(Δy, Δx))
    θ[1] = θ[2]

    # get width function working for short tracks
    wspline = Spline1D(width_values[:, 1], width_values[:, 2] .* 3; k=4)
    wfn(s) = const_width ? 3.0 : wspline((s - s1) / 261)


    return Track(X, Y, Array([X Y]'), curvature, N, P, S_f, S, δs, K, wfn, θ)
end

"""
Create a track from saved waypoints coordinates.

#Arguments
  - `start_waypoint` index of first wp to keep
  - `keep_n_waypoints` can be used to keep only the first N waypoints. Set to -1 to keep all.
  - `resolution` used to upsample track waypoints through interpolation.
"""
function Track(; 
    start_waypoint=2,
    keep_n_waypoints=-1,
    resolution=0.00001,
    npyfile=nothing,
    const_width=false,
    track_length=261,
    )
    # load data
    npyfile = isnothing(npyfile) ? "src/hairpin.npy" : npyfile
    XY = npzread(npyfile)

    # trim waypoints
    start_waypoint = max(2, start_waypoint)
    keep_n_waypoints = if keep_n_waypoints > 0
        min(keep_n_waypoints, size(XY, 1))
    else
        (size(XY, 1) - start_waypoint)
    end

    if isnothing(track_length)
        track_length = (Int64 ∘ round)(sum(sqrt.(diff(XY[:, 1]).^2 .+ diff(XY[:, 2]).^2))) * 2
    end

    end_idx = min(start_waypoint + keep_n_waypoints - 1, track_length)
    XY = XY[start_waypoint:end_idx, :]

    s1 = (start_waypoint) / track_length

    return Track(XY, s1; resolution=resolution, const_width=const_width)
end

"""
Create a `Track` with the initial waypoint closest to the position of 
the bike (defined by its State).
"""
function Track(state; keep_n_waypoints=-1, resolution=0.00001)
    XY = npzread("src/hairpin.npy")
    idx = closest_point_idx(XY[:, 1], state.x, XY[:, 2], state.y)

    return Track(;
        start_waypoint=idx, keep_n_waypoints=keep_n_waypoints, resolution=resolution
    )
end


const FULLTRACK = Track(; start_waypoint=4)

"""
Trim the full track from a start value keeping a given length
"""
function trim(track::Track, svalue, length)
    # svalue = svalue < 1 ? svalue * 259 : svalue
    
    first = findfirst(track.S .>= (svalue))

    last = findlast(track.S .<= (svalue + length))

    return Track(
        FULLTRACK.X[first:last],
        FULLTRACK.Y[first:last],
        FULLTRACK.XY[:, first:last],
        FULLTRACK.curvature[first:last],
        -1,
        FULLTRACK.P[first:last],
        FULLTRACK.S[last],
        FULLTRACK.S[first:last],
        FULLTRACK.δs[first:last],
        FULLTRACK.κ,
        FULLTRACK.width,
        FULLTRACK.θ[first:last],
    )
end