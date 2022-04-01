module forwardmodel
import InfiniteOpt: value, InfiniteModel, supports
using Interpolations: Interpolations
import Parameters: @with_kw

import jcontrol: Track, upsample, int, ξ, closest_point_idx, State
import ..control: KinematicsProblem, DynamicsProblem, MTMproblem
import ..bicycle: State, Bicycle

export Solution, run_forward_model, solution2state

int(x) = (Int64 ∘ round)(x)

@with_kw struct Solution
    δt::Float64
    t::Vector{Float64}    # time
    s::Vector{Float64}    # track progression
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
    Fu::Vector{Float64}

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
    for (ni, i) in enumerate(svalues)
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
    δs = map(ξ(value(model[:δ])), svalues)
    δ̇s = map(ξ(value(model[:δ̇])), svalues)
    us = map(ξ(value(model[:u])), svalues)
    ωs = map(ξ(value(model[:ω])), svalues)
    ns = map(ξ(value(model[:n])), svalues)
    ψs = map(ξ(value(model[:ψ])), svalues)

    if problemtype isa KinematicsProblem
        βs = map(ξ(value(model[:β])), svalues)
        u̇s = map(ξ(value(model[:u̇])), svalues)
    else
        vs = map(ξ(value(model[:v])), svalues)
        βs = map(
                ξ(atan.(vs ./ (us .+ eps()))), svalues
            )
        Fus = map(ξ(value(model[:Fu])), svalues)
    end



    # get values at regular Δt
    Ts = map(ξ(value(model[:t])), svalues)
    time = Ts[1]:δt:Ts[end]
    I() = zeros(Float64, length(time))
    T, X, Y, θ, δ, δ̇, u̇ = I(), I(), I(), I(), I(), I(), I()
    u, ω, n, ψ, β, v, Fu = I(), I(), I(), I(), I(), I(), I()
    for (i, t) in enumerate(time)
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
        β[i] = βs[idx]

        if problemtype isa KinematicsProblem            
            u̇[i] = u̇s[idx]
        else
            v[i] = vs[idx]
            Fu[i] = Fus[idx]
        end
    end

    # get `s`
    S::Vector{Float64} = []
    for i in 1:length(X)
        # get the closest track point
        idx = closest_point_idx(track.X, X[i], track.Y, Y[i])
        push!(S, track.S[idx])
    end 

    return Solution(
        δt=δt,
        t = T,
        s = S,
        x = X,
        y = Y,
        θ = θ,
        u = u,
        ω = ω,
        δ = δ,
        δ̇ = δ̇,
        u̇ = u̇,
        n = n,
        ψ = ψ,
        β = β,
        v = v,
        Fu=Fu,
    )
end


"""
Take the value of a forward model solution at a given svalue
and return a State with the solution values there.
"""
function solution2state(svalue::Float64, solution::Solution)::State
    svalue = svalue < 1 ? svalue * 258 : svalue
    idx = findfirst(solution.s .>= svalue)
    vars =  (:x, :y, :θ, :δ, :δ̇, :ω, :u, :β, :v, :n, :ψ, :Fu)
    return State(;
        Dict(
            map(v -> v=>getfield(solution, v)[idx], vars)
        )...
    )
end

end
