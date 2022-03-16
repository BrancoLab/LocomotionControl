module forwardmodel
import InfiniteOpt: value, InfiniteModel, supports
using DifferentialEquations
import DifferentialEquations: solve as desolve
using Interpolations: Interpolations
import Parameters: @with_kw
import Term.progress: track as pbar

import jcontrol: Track, upsample, kinematics_from_position, int
import ..control: KinematicsProblem, DynamicsProblem
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
    
    # Dynamics problem only
    v::Vector{Float64} = Vector{Float64}[]  # lateral velocity
    ω::Vector{Float64} = Vector{Float64}[]  # angular velocity

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
function run_forward_model(track::Track, model::InfiniteModel; δt=0.01)
    # get model's data and upsample
    n = value(model[:n])
    ψ = value(model[:ψ])
    s = only.(supports(model[:n]))
    Ts = value(model[:t])
    δs = value(model[:δ])
    δ̇s = value(model[:δ̇])
    u̇s = value(model[:u̇])

    n, ψ, s, Ts, δs, δ̇s, u̇s  = upsample(n, ψ, s, Ts, δs, δ̇s, u̇s; δp=0.0005)

    # compute bike position with variable Δt
    Xs, Ys, θs = zeros(length(n)), zeros(length(n)), zeros(length(n))
    for i in pbar(1:length(s); transient=false)
        _n, _ψ = n[i], ψ[i]

        # get closest track waypoint
        closest = argmin(abs.(track.S .- s[i]))

        # get position and orientation
        xp = track.X[closest]
        yp = track.Y[closest]
        θp = track.θ[closest]

        # get position of bike
        if _n == 0
            Xs[i] = xp
            Ys[i] = yp
        else
            norm_angle =  θp + π/2

            Xs[i] = xp + _n * cos(norm_angle)
            Ys[i] = yp + _n * sin(norm_angle)
        end

        # get orientation of bike
        θs[i] = θp + _ψ
    end

    # fix variable Δt
    time = collect(0:δt:Ts[end])
    I() = zeros(length(time))
    T, X, Y, θ, δ, δ̇, u̇ = I(), I(), I(), I(), I(), I(), I()
    for (i, t) in pbar(enumerate(time); transient=false)
        # get next larger timestep
        idx = findfirst(Ts .> t)
      
        T[i] = Ts[idx]
        X[i] = Xs[idx]
        Y[i] = Ys[idx]
        θ[i] = θs[idx]
        δ[i] = δs[idx]
        δ̇[i] = δ̇s[idx]
        u̇[i] = u̇s[idx]
    end

    # compute velocities
    fps = int(1/δt)
    u, ω = kinematics_from_position(X, Y, θ; fps=fps, smooth=true, smooth_wnd=.05)
      
    return Solution(
        δt=δt,
        t = T,
        x = X,
        y = Y,
        θ = θ,
        u = u,
        ω = ω,
        δ = δ,
        δ̇ = δ̇,
        u̇ = u̇,
    )
end

end