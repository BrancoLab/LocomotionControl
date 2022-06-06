module bicycle
import Parameters: @with_kw
import MyterialColors: blue_grey_darker, blue_grey, cyan_dark

import jcontrol: closest_point_idx, Track, euclidean, movingaverage

export Bicycle, State

"""
Stores immutable geometric properties of the bike.
Also stores parameters for drawing.
"""
struct Bicycle
    # geometry
    l_f::Number     # distance from COM to front wheel | cm
    l_r::Number     # distance from rear wheel to COM | cm
    L::Number       # total length | cm
    width::Number   # 'width' of bike. Used to staty within track | cm

    # dynamics
    m::Number       # mass | g
    Iz::Number      # moment of angular inertia | Kg⋅m²
    c::Number       # corenring stiffness

    # for drawing
    wheel_length::Number
    wheel_lw::Number
    wheel_color::String
    front_wheel_color::String
    body_lw::Number
    body_color::String

    function Bicycle(; 
        l_f::Number=3,
        l_r::Number=2,
        width::Number=2,
        m_f=12, 
        m_r=11, 
        c=4e3
        )

        # convert units g->Kg, cm->m
        mfKg = m_f # / 100
        mrKg = m_r # / 100
        lfM = l_f # / 100
        lrM = l_r # / 100

        # compute moment of angular inertia        
        Iz = mfKg * lfM^2 + mrKg * lrM^2

        return new(
            l_f,
            l_r,
            l_f + l_r,
            width,
            m_f + m_r,
            Iz,
            c,
            0.8,                # wheel_length
            16,                 # wheel_lw
            blue_grey_darker,   # wheel_color
            cyan_dark,          # front_wheel_color
            8,                  # body_lw
            blue_grey,          # body_color
        )
    end
end

"""
Represents the state of the bicycle model at a moment in time.
Can be used to pass initial and final condistions to the control
model.
"""
@with_kw mutable struct State
    x::Number = 0  # position
    y::Number = 0
    θ::Number = 0  # orientation
    δ::Number = 0  # steering angle
    δ̇::Number = 0
    ω::Number = 0
    u::Number = 0  # velocity  | fpr DynamicsProblem its the longitudinal velocity component

    # KinematicsProblem only
    β::Number = 0  # slip angle

    # DynamicsProblem only
    v::Number = 0  # lateral velocity
    Fu::Number = 0  # forward force

    # track errors 
    n::Number = 0
    ψ::Number = 0

    # track variables
    track_idx = 0
    s = 0
end

"""
    State(trial::Trial, frame::Int, track::Track)

Get `State` from experimental data at a frame
"""
function State(trial, frame::Int, track::Track; smoothing_window=1, kwargs...)
    if smoothing_window == 1
        _x, _y, _ω, _u, _θ = trial.x, trial.y, trial.ω, trial.u, trial.θ
    else
        _x = movingaverage(trial.x, smoothing_window)
        _y = movingaverage(trial.y, smoothing_window)
        _ω = movingaverage(trial.ω, smoothing_window)
        _u = movingaverage(trial.u, smoothing_window)
        _θ = movingaverage(trial.θ, smoothing_window)
    end

    if _θ[1] < 0
        θ = _θ .+ 2π
    else
        θ = _θ
    end

    # get track errors
    x, y = _x[frame], _y[frame]
    dist = sqrt.((track.X.-x).^2 + (track.Y.-y).^2)
    idx = argmin(dist)

    n = dist[idx]

    # get the sign of n right based on which side of the track the mouse is
    if idx > 2 && idx < length(track.X) - 2
        x1, y1, x2, y2 = track.X[idx-1], track.Y[idx-1], track.X[idx+1], track.Y[idx+1]
        d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
        n *= sign(d)
    end


    ψ = track.θ[idx] - θ[frame]
    # ψ = mod(abs(ψ), 2π) 

    return State(; x=x, y=y, θ=θ[frame], ω=_ω[frame], u=_u[frame], n=n, ψ=ψ, track_idx=idx, s=track.S[idx], kwargs...)
end

"""
    State(solution, frame::Int)

Get `State` from a forward problem solution at Δt from start.
"""
function State(solution, Δt::Float64)
    frame = (Int ∘ round)(Δt / solution.δt)

    return State(;
        x=solution.x[frame],
        y=solution.y[frame],
        θ=solution.θ[frame],
        δ=solution.δ[frame],
        δ̇=solution.δ̇[frame],
        ω=solution.ω[frame],
        u=solution.u[frame],
        β=solution.β[frame],
        v=solution.v[frame],
        n=solution.n[frame],
        ψ=solution.ψ[frame],
        Fu = solution.Fu[frame]
    )
end
end
