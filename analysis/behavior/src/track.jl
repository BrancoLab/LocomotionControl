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
        "Track. \e[2m$(round(track.S_f)) cm long, $(track.N) waypoints. Width: $(track.width)cm\e[0m",
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
    [0.0    1.1]  # start
    [.05    1.0]
    [.07    .95] # first narrow
    [.15    1.0]  # end of frst narrow
    [.25    1.1]  # second curve
    [.4     1.0]  # end of second curve
    [.45    0.95]  # second narrow
    [.55    0.95]  # end of second narrow
    [.6     1.2]  # end of second narrow
    [.67    1.5]  # start of fourth curve
    [.72    1.5]
    [.9     1.1]
    [1      1.1]  # end
]


# ---------------------------------------------------------------------------- #
#                              TRACK CONSTRUCTORS                              #
# ---------------------------------------------------------------------------- #

"""
Construct `Track` out of a set of waypoints
"""
function Track(XY, s1::Float64; resolution=0.00001)
    # get new points locations thorugh interpolation
    X, Y = upsample(XY[:, 1], XY[:, 2]; δp = resolution)
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
    wspline = Spline1D(width_values[:, 1], width_values[:, 2] .* 3, k=2)
    wfn(s) = wspline((s - s1)/261)

    # return Track
    return Track(X, Y, Array([X Y]'), curvature, N, P, S_f, S, δs, K, wfn, θ)
end


"""
Create a track from saved waypoints coordinates.

#Arguments
  - `start_waypoint` index of first wp to keep
  - `keep_n_waypoints` can be used to keep only the first N waypoints. Set to -1 to keep all.
  - `resolution` used to upsample track waypoints through interpolation.
"""
function Track(; start_waypoint=1, keep_n_waypoints=-1, resolution=0.00001)
    # load data
    XY = npzread("src/hairpin.npy")

    # trim waypoints
    start_waypoint = max(2, start_waypoint)
    keep_n_waypoints =
        keep_n_waypoints > 0 ? min(keep_n_waypoints, size(XY, 1)) : (size(XY, 1) - start_waypoint)
    XY = XY[start_waypoint:(start_waypoint + keep_n_waypoints - 1), :]

    s1 = (start_waypoint)/261

    return Track(XY, s1; resolution=resolution)
end

"""
Create a `Track` with the initial waypoint closest to the position of 
the bike (defined by its State).
"""
function Track(state; keep_n_waypoints=-1, resolution=0.00001)
    XY = npzread("src/hairpin.npy")
    idx = closest_point_idx(XY[:, 1], state.x, XY[:, 2], state.y)

    return Track(; start_waypoint=idx, keep_n_waypoints=keep_n_waypoints, resolution=resolution)
end