
using Statistics: mean


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