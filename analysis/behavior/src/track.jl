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

# ---------------------------------------------------------------------------- #
#                                track creation                                #
# ---------------------------------------------------------------------------- #
"""
Ugly hard coded function to change track width as a function of curvilinear
distance along the track.
"""
function widtfn(s, S_f, width) 
    r = (s/S_f) 
    if r < .07
        fact = 1

    elseif r < .185
        fact = .7

    elseif r < .21
        fact = 1

    elseif r < .44
        fact = 1.25

    elseif r < .46
        fact = 1

    elseif r < .55
        fact = .7

    elseif r < .58
        fact = 1

    elseif r < .6
        fact = 1.25

    # elseif r < .66
    #     fact = 2

    else
        fact = 1.5
    end
    return fact * width
end



"""
Create a track from saved waypoints coordinates.

#Arguments
  - `keep_n_waypoints` can be used to keep only the first N waypoints. Set to -1 to keep all.
  - `δ` can be used to select every δ waypoints (downsample)
  - `resolution` used to upsample track waypoints through interpolation.
"""
function Track(; keep_n_waypoints=-1, δ=1, resolution=0.005)
    # load data
    XY = npzread("src/hairpin.npy")
    keep_n_waypoints =
        keep_n_waypoints > 0 ? min(keep_n_waypoints, size(XY, 1)) : size(XY, 1)
    XY = XY[1:δ:keep_n_waypoints, :]

    # get new points locations thorugh interpolation
    t = range(0, 1; length=length(XY[:, 1]))
    x = XY[:, 1]
    y = XY[:, 2]
    A = hcat(x, y)

    itp = Interpolations.scale(
        interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t, 1:2
    )

    tfine = 0:resolution:1
    X, Y = [itp(t, 1) for t in tfine], [itp(t, 2) for t in tfine]

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
    θ = atan.(Δy, Δx)
    θ[1] = θ[2]

    return Track(X, Y, Array([X Y]'), curvature, N, P, S_f, S, δs, K, (s)->widtfn(s, S_f, 3), θ)
end
