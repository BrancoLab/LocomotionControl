"""
Compute velocity `u` and angular velocity `ω` from position.

Assumes θ in radians.
Assumes values at equally spaced time intervals.
"""
function kinematics_from_position(
    x::Vector{Float64},
    y::Vector{Float64},
    θ::Vector{Float64},
    speed;
    fps::Int=60,
    smooth::Bool=false,
    smooth_wnd::Float64=0.1,  # smoothing window size in seconds
)
    # compute speed
    # speed = sqrt.(Δ(x) .^ 2 .+ Δ(y) .^ 2) .* fps

    # compute longitudinal and lateral velocities
    β = atan.(Δ(y), (Δ(x) .+ eps())) .- θ
    u = @. speed * cos(β)
    v = @. speed * sin(β)

    # compute angular velocity
    time = (collect(0:length(speed)) ./ fps)[2:end]
    ω = ∂(time, unwrap(θ))

    # smooth
    if smooth
        wnd = int(fps * smooth_wnd)
        speed = movingaverage(speed, wnd)
        u = movingaverage(u, wnd)
        v = movingaverage(v, wnd)
        ω = movingaverage(ω, wnd)
    end

    return speed, u, v
end
