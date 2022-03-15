using Statistics: mean
using ForwardDiff

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

"""
Interpolate an array and get the derivative at
time values.

From: https://discourse.julialang.org/t/differentiation-without-explicit-function-np-gradient/57784
"""
function ∂(time, x)
    itp = interpolate((time,), x, Gridded(Linear()));
    return map((t)->ForwardDiff.derivative(itp, t), time)
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

"""
Simple linear interpolation
"""
ξ(x) = interpolate(x, BSpline(Linear()))

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
      while unwrapped[i] - unwrapped[i-1] >= π
        unwrapped[i] -= 2π
      end
      while unwrapped[i] - unwrapped[i-1] <= -π
        unwrapped[i] += 2π
      end
    end
    return unwrapped
  end
  
unwrap!(v) = unwrap(v, true)

"""
Moving average with window size `n`
"""
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]