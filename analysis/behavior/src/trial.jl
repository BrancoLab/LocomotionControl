module trial
import Parameters: @with_kw
import DataFrames: DataFrameRow, DataFrame
import JSONTables: jsontable

import jcontrol: Track, closest_point_idx, unwrap, kinematics_from_position, movingaverage
import ..comparisons: TrackSegment

export Trial, trimtrial, get_varvalue_at_frameidx

"""
Compute values of `s` based on position along a trial
"""
function get_trial_s(trial::DataFrameRow, track::Track)
    x, y = trial.body_x, trial.body_y
    n = length(x)

    S::Vector{Float64} = []
    for i in 1:n
        # get the closest track point
        idx = closest_point_idx(track.X, x[i], track.Y, y[i])
        push!(S, track.S[idx])
    end

    return S
end

"""
Store trial information
"""
struct Trial
    x::Vector{Float64}
    y::Vector{Float64}
    s::Vector{Float64}
    θ::Vector{Float64}
    ω::Vector{Float64}
    speed::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
    duration::Float64
end

Base.show(io::IO, trial::Trial) = print(io, "Trial: $(round(trial.duration; digits=2))s")

"""
Construct Trial out of a dataframe entry.
"""
function Trial(trial::DataFrameRow, track::Track; fixstart=true)
    s = get_trial_s(trial, track)

    # get orientation
    θ = unwrap(
        atan.(trial.snout_y .- trial.tail_base_y, trial.snout_x .- trial.tail_base_x)
    )
    θ = movingaverage(θ, 30)

    # get velocities
    speed, u, v, ω = kinematics_from_position(
        trial.body_x,
        trial.body_y,
        θ,
        trial.body_speed;
        fps=60,
        smooth=true,
        smooth_wnd=0.05,
    )

    # remove artifacts
    # ω = ω
    # ω[abs.(ω) .> 15] .= 0

    # trim start to when speed is high enough
    if fixstart
        start = findfirst(speed .> 15)
        start = isnothing(start) ? 1 : start
    else
        start = 1
    end

    return Trial(
        trial.body_x[start:end],    # x
        trial.body_y[start:end],    # y
        s[start:end],               # s
        θ[start:end],               # θ
        ω[start:end],               # ω
        speed[start:end],           # speed
        u[start:end],               # u
        v[start:end],               # v
        trial.duration,             # duration
    )
end

function Trial(filepath::String)
    open(filepath) do f
        data = jsontable(read(f))
        return Trial(
            data.x,             # x
            data.y,             # y
            data.s,             # s
            data.θ,             # θ
            data.ω,             # ω
            data.speed,         # speed
            data.u,             # u
            data.v,             # v
            data.duration[1],    # duration
        )
    end
end

# ----------------------------------- utils ---------------------------------- #
"""
Cut a trial to keep only the data between two s-values.
"""
function trimtrial(trial::Trial, s0, s1; by=:space)
    if by == :space
        start = findfirst(trial.s .>= s0)
        isnothing(start) && return nothing

        stop = findlast(trial.s .<= s1)
        isnothing(stop) && return nothing
    else
        start, stop = s0, s1
    end

    return Trial(
        trial.x[start:stop],        # x
        trial.y[start:stop],        # y
        trial.s[start:stop],        # s
        trial.θ[start:stop],        # θ
        trial.ω[start:stop],        # ω
        trial.speed[start:stop],    # speed
        trial.u[start:stop],        # u
        trial.v[start:stop],        # v
        (stop - start) / 60,            # duration
    )
end

function trimtrial(trial::Trial, seg::TrackSegment)
    return trimtrial(trial, 260 * (seg.s₀ - 0.01), 260 * (seg.s₁ - 0.01))
end

"""
    Get the value of each trial's variable at a frame index
"""
function get_varvalue_at_frameidx(trials::Vector{Trial}, variable::Symbol, frame::Int)
    return map(t -> getfield(t, variable)[frame], trials)
end

"""
    Get the value of each trial's variable at the last frame
"""
function get_varvalue_at_frameidx(trials::Vector{Trial}, variable::Symbol)
    return map(t -> getfield(t, variable)[end], trials)
end

end
