module forwardmodel
import InfiniteOpt: value, InfiniteModel, supports
using Interpolations: Interpolations
import Parameters: @with_kw
import Term.progress: track as pbar

import jcontrol: Track, upsample, kinematics_from_position, int, ξ
import ..control: KinematicsProblem, DynamicsProblem, MTMproblem
import ..bicycle: State, Bicycle

export Solution, run_forward_model

int(x) = (Int64 ∘ round)(x)

@with_kw struct Solution
    δt::Float64
    t::Vector{Float64}    # time
    x::Vector{Float64}    # position
    y::Vector{Float64}
    θ::Vector{Float64}    # orientation
    δ::Vector{Float64}    # steering angle
    u::Vector{Float64}    # velocity  | fpr DynamicsProblem its the longitudinal velocity component
    ω::Vector{Float64} = Vector{Float64}[]  # angular velocity

    n::Vector{Float64}    # track errors
    ψ::Vector{Float64}

    # Kinematic problem only
    β::Vector{Float64}

    # Dynamics problem only
    v::Vector{Float64} = Vector{Float64}[]  # lateral velocity

    # controls
    u̇::Vector{Float64}
    δ̇::Vector{Float64}
end

"""
To get the bike's trajectory in time and wrt the allocentric
track's reference frame (instead of curvilinear coordinates),
this function takes the error `n` and `ψ` and computes the bike
position wrt to the track's centerline.

Since `n` and `ψ` are defined wrt to s, first the bike's position
is computed at increments of s, then this is upsampled and converted
to the corresponding time increments.

Finally, the bike's velocity and angular velocities are computed.
"""
function run_forward_model(
    problemtype::MTMproblem, track::Track, model::InfiniteModel; δt=0.01
)
    # get model's data and upsample
    n = ξ(value(model[:n]))
    ψ = ξ(value(model[:ψ]))
    s = ξ(value(model[:s]))

    # compute bike position with variable Δt (wrt s)
    svalues = 1:0.25:length(value(model[:n]))
    II() = zeros(Float64, length(svalues))
    Xs, Ys, θs = II(), II(), II()
    for (ni, i) in pbar(enumerate(svalues); redirectstdout=false)
        _n, _ψ = n[i], ψ[i]

        # get closest track waypoint
        closest = argmin(abs.(track.S .- s[i]))

        # get position and orientation of track
        xp = track.X[closest]
        yp = track.Y[closest]
        θp = track.θ[closest]

        # get position of bike
        if _n == 0
            Xs[ni] = xp
            Ys[ni] = yp
        else
            norm_angle = θp + π / 2

            Xs[ni] = xp + _n * cos(norm_angle)
            Ys[ni] = yp + _n * sin(norm_angle)
        end

        # get orientation of bike
        θs[ni] = θp + _ψ
    end

    # get other variables
    Ts = map(ξ(value(model[:t])), svalues)
    δs = map(ξ(value(model[:δ])), svalues)
    δ̇s = map(ξ(value(model[:δ̇])), svalues)
    us = map(ξ(value(model[:u])), svalues)
    ωs = map(ξ(value(model[:ω])), svalues)
    ns = map(ξ(value(model[:n])), svalues)
    ψs = map(ξ(value(model[:ψ])), svalues)

    if problemtype isa KinematicsProblem
        βs = map(ξ(value(model[:β])), svalues)
        u̇s = map(ξ(value(model[:u̇])), svalues)
    end

    # get values at regular Δt
    time = Ts[1]:δt:Ts[end]
    I() = zeros(Float64, length(time))
    T, X, Y, θ, δ, δ̇, u̇ = I(), I(), I(), I(), I(), I(), I()
    u, ω, n, ψ, β = I(), I(), I(), I(), I()
    for (i, t) in pbar(enumerate(time); redirectstdout=false)
        idx = findfirst(Ts .>= t)
        idx = isnothing(idx) ? 1 : idx

        T[i] = Ts[idx]
        X[i] = Xs[idx]
        Y[i] = Ys[idx]
        θ[i] = θs[idx]
        δ[i] = δs[idx]
        δ̇[i] = δ̇s[idx]
        u[i] = us[idx]
        ω[i] = ωs[idx]
        n[i] = ns[idx]
        ψ[i] = ψs[idx]

        if problemtype isa KinematicsProblem
            β[i] = βs[idx]
            u̇[i] = u̇s[idx]
        end
    end

    return Solution(; δt=δt, t=T, x=X, y=Y, θ=θ, u=u, ω=ω, δ=δ, δ̇=δ̇, u̇=u̇, n=n, ψ=ψ, β=β)
end

end
