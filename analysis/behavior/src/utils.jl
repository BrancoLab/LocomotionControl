
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