using Statistics: mean
using ForwardDiff
using Dierckx

function toDict(res)
    return Dict(
        fieldnames(typeof(res)) .=> getfield.(Ref(res), fieldnames(typeof(res)))
    )
end


function naturalsort(x::Vector{String})
    f = text -> all(isnumeric, text) ? Char(parse(Int, text)) : text
    sorter = key -> join(f(c) for c in eachmatch(r"[0-9]+|[^0-9]+", key))
    sort(x, by=sorter)
end


# ------------------------------ interpolations ------------------------------ #

"""
Simple linear interpolation
"""
ξ(x) = interpolate(x, BSpline(Linear()))

"""
  interpolate_wrt_to(x, y)

Interpolate `y` with respect to `x`.

Given arrays `a=1:101` and `time=0:100`, `ξ(a)` gives an interpolation
object that can only be queried by values ∈ [1, 101] (the indices range of `a`).
`interpolate_wrt_to(time, a)` on the other hand gives an interpolation object that
can be queried by values ∈ [0, 100], the values ranges of `time`.
"""
function interpolate_wrt_to(x, y)
    @assert length(x) == length(y) "The x and data array should have the same length"
    return Spline1D(x, y; w=ones(length(x)), k=1, bc="nearest", s=0.0)
end

"""
Upsamble a set of variables through interpolation
"""
function upsample(data...; δp=0.001)
    n = length(data)
    P = range(0, 1, length(data[1]))

    # upsample data
    itp = Interpolations.scale(
        interpolate(hcat(data...), (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), P, 1:n
    )

    tfine = 0:δp:1
    upsampled = []
    for i in 1:n
        push!(upsampled, [itp(t, i) for t in tfine])
    end
    return upsampled
end

# -------------------------------- derivatives ------------------------------- #
"""
Interpolate an array and get the derivative at
time values.

From: https://discourse.julialang.org/t/differentiation-without-explicit-function-np-gradient/57784
"""
function ∂(time, x)
    itp = interpolate((time,), x, Gridded(Linear()))
    return map((t) -> ForwardDiff.derivative(itp, t), time)
end

"""
Prepend a 0 to a vector, useful when takin diff
but want the resulting vector to have the right length
"""
ρ(x) = vcat([0], x)

"""
Shorthand for diff givin a vector of right length
"""
Δ(x) = ρ(diff(x))

# ----------------------------------- misc ----------------------------------- #
"""
Round a Float and turn it into an integer
"""
int(x::Number) = (Int ∘ round)(x)

""" 
Unwrap circular variables (in radians)
"""
function unwrap(v, inplace=false)
    # currently assuming an array
    unwrapped = inplace ? v : copy(v)
    for i in 2:length(v)
        while unwrapped[i] - unwrapped[i - 1] >= π
            unwrapped[i] -= 2π
        end
        while unwrapped[i] - unwrapped[i - 1] <= -π
            unwrapped[i] += 2π
        end
    end
    return unwrapped
end

unwrap!(v) = unwrap(v, true)

# --------------------------------- smoothing -------------------------------- #
"""
Moving average with window size `n`
"""
function movingaverage(g, n)
    return [i < n ? mean(g[begin:i]) : mean(g[(i - n + 1):i]) for i in 1:length(g)]
end

# --------------------------------- geometry --------------------------------- #

"""
  closest_point_idx(X::Vector{Number}, x::Number, Y::Vector{Number}, y::Number)

Get the index of the point in a vector (X, Y) closest to a point (x,y)
"""
function closest_point_idx(X::Vector{T}, x::T, Y::Vector{T}, y::T) where {T<:Number}
    return argmin(sqrt.((X .- x) .^ 2 .+ (Y .- y) .^ 2))
end

euclidean(x0, x1, y0, y1) = √((x0 - x1)^2 + (y0 - y1)^2)
