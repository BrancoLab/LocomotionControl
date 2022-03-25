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
@with_kw struct Trial
    x::Vector{Float64}
    y::Vector{Float64}
    s::Vector{Float64}
    θ::Vector{Float64} = Vector{Float64}[]
    ω::Vector{Float64} = Vector{Float64}[]
    u::Vector{Float64} = Vector{Float64}[]
    duration::Float64
end

"""
Construct Trial out of a dataframe entry.
"""
function Trial(trial::DataFrameRow, track::Track)
    s = get_trial_s(trial, track)

    # get orientation
    θ = unwrap(
        atan.(trial.snout_y .- trial.tail_base_y, trial.snout_x .- trial.tail_base_x)
    )
    θ = movingaverage(θ, 3)

    # get velocities
    u, ω = kinematics_from_position(
        trial.body_x, trial.body_y, θ; fps=60, smooth=true, smooth_wnd=0.05
    )

    # remove artifacts
    ω = ω
    ω[abs.(ω) .> 15] .= 0

    # trim start to when speed is high enough
    start = findfirst(u .> 25)

    return Trial(;
        x=trial.body_x[start:end],
        y=trial.body_y[start:end],
        s=s[start:end],
        θ=θ[start:end],
        ω=ω[start:end],
        u=u[start:end],
        duration=trial.duration,
    )
end


function Trial(filepath::String)
    open(filepath) do f
        data = jsontable(read(f))
        return Trial(;
                x=data.x,
                y=data.y,
                s=data.s,
                θ=data.θ,
                ω=data.ω,
                u=data.u,
                duration=data.duration[1]
            )
    end
end


# ----------------------------------- utils ---------------------------------- #
"""
Cut a trial to keep only the data between two s-values.
"""
function trimtrial(trial::Trial, s0, s1)
    start = findfirst(trial.s .>= s0)
    isnothing(start) && return nothing

    stop = findlast(trial.s .<= s1)
    isnothing(start) && return nothing

    return Trial(
        x=trial.x[start:stop],
        y=trial.y[start:stop],
        s=trial.s[start:stop],
        θ=trial.θ[start:stop],
        ω=trial.ω[start:stop],
        u=trial.u[start:stop],
        duration=-1.0,
    )
end
trimtrial(trial::Trial, seg::TrackSegment) = trimtrial(trial,260 * ( seg.s₀ - .01), 260 * (seg.s₁ - .01))


"""
    Get the value of each trial's variable at a frame index
"""
get_varvalue_at_frameidx(trials::Vector{Trial}, variable::Symbol, frame::Int) = map(t->getfield(t, variable)[frame], trials)

"""
    Get the value of each trial's variable at the last frame
"""
get_varvalue_at_frameidx(trials::Vector{Trial}, variable::Symbol) = map(t->getfield(t, variable)[end], trials)


end