using Plots
using Statistics: mean
import MyterialColors: blue_grey_darker, blue_dark, green
import InfiniteOpt: termination_status
using CSV
import Colors: HSL
import DataFrames: DataFrame
using Term.progress
import Term.progress: with as withpbar
import Term: install_term_logger, install_stacktrace
import Printf: @sprintf
install_term_logger()
install_stacktrace()

using jcontrol
import jcontrol.comparisons: track_segments, TrackSegment
using jcontrol.visuals
import jcontrol: trimtrial, euclidean, FULLTRACK, Track, trim, toDict, ControlOptions
import jcontrol.Run: run_mtm
import jcontrol.forwardmodel: solution2state, solution2s
import jcontrol.io: PATHS


@Base.kwdef mutable struct SimTracker
    s0::Float64      = 0
    s::Float64      = 0
    iter::Int       = 0
    t::Float64      = 0.0
    Δt::Float64     = 0.01
    prevsol         = nothing
    hasfailed       = 0
end

@Base.kwdef mutable struct SolutionTracker
    t::Vector{Float64} = Vector{Float64}[]      
    s::Vector{Float64} = Vector{Float64}[]      
    x::Vector{Float64} = Vector{Float64}[]      
    y::Vector{Float64} = Vector{Float64}[]  
    θ::Vector{Float64} = Vector{Float64}[]      
    δ::Vector{Float64} = Vector{Float64}[]      
    u::Vector{Float64} = Vector{Float64}[]      
    ω::Vector{Float64} = Vector{Float64}[]  
    n::Vector{Float64} = Vector{Float64}[]      
    ψ::Vector{Float64} = Vector{Float64}[]  
    β::Vector{Float64} = Vector{Float64}[]  
    v::Vector{Float64} = Vector{Float64}[]
    Fu::Vector{Float64} = Vector{Float64}[]  
    δ̇::Vector{Float64} = Vector{Float64}[]  
end

function add!(tracker::SolutionTracker, t::Float64, state::State, s::Float64)
    push!(tracker.t, t)

    for v in (:y, :x, :θ, :δ, :u, :ω, :n, :ψ, :β, :v, :Fu, :δ̇)
        push!(getfield(tracker, v), getfield(state, v)[1])
    end

    push!(tracker.s, s)
end

"""
take states at Δt interval while 0 < t < tf and store in tracker
"""
function extend_with_sol(tracker::SolutionTracker, sol::Solution, Δt::Float64, tf::Float64)
    Δt > tf && return

    for t in 0:Δt:tf
        state = solution2state(t, sol; at=:time)
        s = solution2s(t, sol)
        add!(tracker, t, state, s)
    end
end

"""
Repeate MTM with windows of decreasing length
"""
function attempt_step(simtracker, control_model, s0, initial_state, planning_horizon)
    @warn "Could not solve $(simtracker.iter)" termination_status(control_model) s0

    # try again , either shortening or lengthening the planning window
    solution = nothing
    for i in 1:6
        _sign  = i < 3 ? +1 : -1
        len = sqrt(initial_state.v^2 + initial_state.u^2) * (planning_horizon + (_sign * .01 * i))
        if len < 4
            @warn "Length is $len<4"
            skip=true
            len = 4
        else
            skip = false
        end
        track = trim(FULLTRACK, s0, len)

        _, _, control_model, solution = run_mtm(
            :dynamics,
            1.5;
            track=track,
            icond=initial_state,
            fcond=:minimal,
            control_options=:default,
            showplots=false,
            n_iter=5000,
            quiet=true,
        )

        converged(control_model) && break
        skip && break
    end
    return converged(control_model), solution
end

converged(control_model) = "LOCALLY_SOLVED" == string(termination_status(control_model))
fail() = nothing, nothing, true, nothing


"""
Perform a simulation step
"""
function step(simtracker, globalsolution, planning_horizon::Float64,)
    # get initial conditions
    if simtracker.iter == 1
        initial_state =  solution2state(simtracker.s0, globalsolution)
        simtracker.s = simtracker.s0
        simtracker.t += simtracker.Δt
    else
        initial_state =  solution2state(simtracker.Δt, simtracker.prevsol; at=:time)
        simtracker.s = solution2s(simtracker.Δt, simtracker.prevsol)

        idx = findfirst(simtracker.prevsol.t .>= simtracker.Δt)
        idx = isnothing(idx) ? length(simtracker.prevsol.t) : idx
        simtracker.t += simtracker.prevsol.t[idx]
    end


    # get planning window (trimmed track) & final conditions
    s0 =  max(0.001, simtracker.s)
    if s0 < 250
        len = max(sqrt(initial_state.v^2 + initial_state.u^2) * planning_horizon, 4)
        track = trim(FULLTRACK, s0, len)
        final_state = nothing
    else
        track = trim(FULLTRACK, s0, 200)
        final_state = State(; u=30, ω=0)
    end

    # (attempt to) solve MTM problem over planning window
    control_model, solution = nothing, nothing
    try
        _, _, control_model, solution = run_mtm(
            :dynamics,
            1.5;
            track=track,
            icond=initial_state,
            fcond=final_state,
            control_options=:default,
            showplots=false,
            n_iter=5000,
            quiet=true,
        )
    catch
        return fail()
    end

    # failed -> try alternative strategies
    if !converged(control_model)
        # try to recover a solution by shortening the planning window
        success, solution = attempt_step(simtracker, control_model, s0, initial_state, planning_horizon)
        success || return fail()
    end
    simtracker.prevsol = solution
    
    return initial_state, solution, false, track
end

"""
Run a simulation in which the model can only plan for `planning_horizon` seconds ahead.
"""
function run_simulation(; s0=0.0, sf=258, planning_horizon::Float64=.5, n_iter=250, Δt=.01)
    # run global solution
    track = Track(;start_waypoint=4, keep_n_waypoints=-1)

    _, bike, _, globalsolution = run_mtm(
        :dynamics,
        1.5;
        showtrials=nothing,
        track=track,
        n_iter=5000,
        fcond=State(; u=30, ω=0),
        timed=false,
        showplots=false,
    )

    # plot background & global trajectory
    colors = [
            range(HSL(colorant"red"), stop=HSL(colorant"green"), length=(Int ∘ floor)(n_iter/2))...,
            range(HSL(colorant"green"), stop=HSL(colorant"blue"), length=(Int ∘ ceil)(n_iter/2))...
    ]

    simtracker = SimTracker(s0=s0, Δt=Δt)
    solutiontracker = SolutionTracker()
    pbar = ProgressBar(; expand=true, columns=:detailed)
    job = addjob!(pbar; N=n_iter, description="Running horizon: $planning_horizon seconds")

    data = nothing
    withpbar(pbar) do
        anim = @animate for i in 1:n_iter
            simtracker.iter = i

            # run simulation and store results
            initial_state, solution, shouldstop, track = step(simtracker, globalsolution, planning_horizon)
            shouldstop && begin
                extend_with_sol(solutiontracker, simtracker.prevsol, Δt, planning_horizon)
                break
            end
            add!(solutiontracker, simtracker.t, initial_state, simtracker.s)

            # plot stuff
            p1 = draw(:arena)
            plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_grey_darker, lw=6, alpha=.8, label=nothing)
            draw!(FULLTRACK; border_alpha=.25, alpha=0.0)

            draw!(track; color=colors[i], border_lw=5, alpha=0.0)
            plot_bike_trajectory!(solution, bike; showbike=false, label=nothing, color=colors[i], alpha=.8, lw=4)
            draw!(initial_state; color=colors[i], alpha=1)
            plot!(; title="T: $(round(simtracker.t; digits=2))s | horizon: $planning_horizon s")

            simtracker.s > sf && break
            update!(job)
            sleep(0.001)
        end

        @info "SAVING ANIMATIONS"
        # save animation
        name = (@sprintf "s0_%.2f_horizon_length_%.2f" s0 planning_horizon)
        gifpath = joinpath(PATHS["horizons_sims_cache"], "$name.gif")
        gif(anim, gifpath, fps=(Int ∘ round)(0.2/Δt))

        # save video
        vidpath = joinpath(PATHS["horizons_sims_cache"], "$name.mp4")
        mp4(anim, vidpath, fps=(Int ∘ round)(0.2/Δt))

        # save global solution
        destination = joinpath(PATHS["horizons_sims_cache"], "global_solution.csv")
        data = DataFrame(toDict(globalsolution))
        CSV.write(destination, data)

        # save short horizon solution
        destination = joinpath(PATHS["horizons_sims_cache"], "$name.csv")
        data = DataFrame(toDict(solutiontracker))
        CSV.write(destination, data)
    end


    return data
end


horizons = vcat(collect(.08:.01:.4), collect(.45:.05:1.2))
starts = [0.0, 45.0, 98.0, 150.0]
ends = [38.0, 96.0, 142.0, 220.0]

for i in 2:length(starts)
    for horizon in horizons
        @info "Running horizon length $horizon seconds on curve $i"
        results = run_simulation(planning_horizon=horizon, s0=starts[i], sf=ends[i])
        # break
    end

end

