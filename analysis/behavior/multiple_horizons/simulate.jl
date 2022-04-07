using Plots
using Statistics: mean
import MyterialColors: blue_grey_darker, blue_dark, green
import InfiniteOpt: termination_status
using CSV
import Colors: HSL
import DataFrames: DataFrame
import Term: track as pbar
import Term: install_term_logger
import Printf: @sprintf
install_term_logger()

using jcontrol
import jcontrol.comparisons: track_segments, TrackSegment
using jcontrol.visuals
import jcontrol: trimtrial, euclidean, FULLTRACK, Track, trim, toDict, ControlOptions
import jcontrol.Run: run_mtm
import jcontrol.forwardmodel: solution2state, solution2s
import jcontrol.io: PATHS


extreme_coptions = ControlOptions(;
    u_bounds=Bounds(10, 80),
    δ_bounds=Bounds(-110, 110, :angle),
    δ̇_bounds=Bounds(-6, 6),
    ω_bounds=Bounds(-600, 600, :angle),
    v_bounds=Bounds(-30, 30),
    Fu_bounds=Bounds(-1200, 3500),
)

@Base.kwdef mutable struct SimTracker
    s::Float64      = 0
    iter::Int       = 0
    t::Float64      = 0.0
    Δt::Float64     = 0.01
    prevsol         = nothing
    prevvalidsol    = nothing
    nskipped::Int   = 0
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


function step(simtracker, globalsolution, planning_horizon::Float64)
    
    # trim track to define planning window
    # get the initial state
    if simtracker.iter == 1
        initial_state = globalsolution[1]
        simtracker.s = 0.0
        simtracker.t += simtracker.Δt
    else
        initial_state =  solution2state(simtracker.Δt, simtracker.prevsol; at=:time)
        simtracker.s = solution2s(simtracker.Δt, simtracker.prevsol)

        idx = findfirst(simtracker.prevsol.t .>= simtracker.Δt)
        idx = isnothing(idx) ? length(simtracker.prevsol.t) : idx
        simtracker.t += simtracker.prevsol.t[idx]
    end


    # get where the previous simulation was at planning_horizon
    s0 =  max(0.001, simtracker.s)
    if s0 < 220
        len = max(sqrt(initial_state.v^2 + initial_state.u^2) * planning_horizon, 5)
        track = trim(FULLTRACK, s0, len)
        final_state = nothing
    else
        track = trim(FULLTRACK, s0, 200)
        final_state = State(ψ=0.0, u=25, ω=0.0)
    end

    # fit model
    control_model, solution = nothing, nothing
    try
        _, _, control_model, solution = run_mtm(
            :dynamics,
            3;
            track=track,
            icond=initial_state,
            fcond=final_state,
            control_options=:default,
            showplots=false,
            n_iter=5000,
            quiet=true,
        )

    catch
        return nothing, nothing, true
    end

    if "LOCALLY_SOLVED" != string(termination_status(control_model))
        @warn "Could not solve $(simtracker.iter)" termination_status(control_model) s0
        success = false

        # try again but with different control options
        # _, _, control_model, solution = run_mtm(
        #     :dynamics,
        #     3;
        #     control_options=extreme_coptions,
        #     track=track,
        #     icond=initial_state,
        #     fcond=final_state,
        #     showplots=false,
        #     n_iter=5000,
        #     quiet=true,
        # )
        # @info "With extreme options" termination_status(control_model)
        # if "LOCALLY_SOLVED" != string(termination_status(control_model))
        #     return nothing, nothing, true
        # end

        simtracker.prevsol = solution
        simtracker.nskipped += 1
        (s0 < 30 || 80 < s0 < 160 || s0 > 220) &&  return nothing, nothing, true
    else
        success = true
        simtracker.prevvalidsol = solution
        simtracker.prevsol = solution
        simtracker.nskipped = 0
    end
    
    return initial_state, solution, false
end

"""
Run a simulation in which the model can only plan for `planning_horizon` seconds ahead.
"""
function run_simulation(; planning_horizon::Float64=.5, n_iter=200, Δt=.025)
    # run global solution
    _, bike, _, globalsolution = run_mtm(
        :dynamics,
        3;
        fcond=State(ψ=0.0, u=25, ω=0.0),
        showplots=false,
        n_iter=5000,
    )

    # plot background & global trajectory
    p1 = draw(:arena)
    plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_grey_darker, lw=6, alpha=.8, label=nothing)
    colors = [
            range(HSL(colorant"red"), stop=HSL(colorant"green"), length=(Int ∘ floor)(n_iter/2))...,
            range(HSL(colorant"green"), stop=HSL(colorant"blue"), length=(Int ∘ ceil)(n_iter/2))...
    ]

    simtracker = SimTracker(Δt=Δt)
    solutiontracker = SolutionTracker()
    anim = @animate for i in pbar(1:n_iter, redirectstdout=false)
        simtracker.iter = i

        # run simulation and store results
        initial_state, solution, shouldstop = step(simtracker, globalsolution, planning_horizon)
        shouldstop && break
        add!(solutiontracker, simtracker.t, initial_state, simtracker.s)

        # plot stuff
        plot_bike_trajectory!(solution, bike; showbike=false, label=nothing, color=colors[i], alpha=.8, lw=4)
        draw!(initial_state; color=colors[i], alpha=1)
        plot!(; title="Iter $i, time: $(round(simtracker.t; digits=2))s")

        simtracker.s > 258 && break
    end

    # save animation
    name = @sprintf "multiple_horizons_mtm_horizon_length_%.2f" planning_horizon
    gifpath = joinpath(PATHS["horizons_sims_cache"], "$name.gif")
    gif(anim, gifpath, fps=(Int ∘ round)(0.2/Δt))

    # save global solution
    destination = joinpath(PATHS["horizons_sims_cache"], "global_solution.csv")
    data = DataFrame(toDict(globalsolution))
    CSV.write(destination, data)

    # save short horizon solution
    destination = joinpath(PATHS["horizons_sims_cache"], "$name.csv")
    data = DataFrame(toDict(solutiontracker))
    CSV.write(destination, data)
    return data
end


done= (.1, .15, .2, .25, .4, .5, .75)
# todo = (25,)

# todo = (.1, .15, .2, .25, .4, .5, .75, 1.0, 1.5, 2.0)
todo = (.10, .15, .20, .25, .40, .50,)

for horizon in todo
    @info "Running horizon length $horizon seconds"
    results = run_simulation(planning_horizon=horizon)
end