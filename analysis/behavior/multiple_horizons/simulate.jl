using Plots
using Statistics: mean
import MyterialColors: blue_grey_dark, blue_dark, green
import InfiniteOpt: termination_status
import Colors: HSL
import Term: track as pbar
import Term: install_term_logger
install_term_logger()

using jcontrol
import jcontrol.comparisons: track_segments, TrackSegment
using jcontrol.visuals
import jcontrol: trimtrial, euclidean, FULLTRACK, Track, trim
import jcontrol.Run: run_mtm
import jcontrol.forwardmodel: solution2state, solution2s

HORIZONS_LENGTH = 15  # cm

RUN_GLOBAL = false
RUN_LOCAL = true


@Base.kwdef mutable struct SimTracker
    s::Float64      = 0
    iter::Int       = 0
    t::Float64      = 0.0
    Δt::Float64     = 0.01
    prevsol         = nothing
end


function step(simtracker, globalsolution)
    simtracker.t += simtracker.Δt

    # trim track to define planning window
    initial_state = simtracker.iter == 1 ? 
                        globalsolution[1] :
                        solution2state(simtracker.Δt, simtracker.prevsol; at=:time)
    simtracker.s = simtracker.iter == 1 ? 0.0 : solution2s(simtracker.Δt, simtracker.prevsol)
    track = trim(FULLTRACK, max(1.0, simtracker.s + .1), HORIZONS_LENGTH)


    # get the final state
    # final_state =  solution2state(
    #                     track.S[end-1], 
    #                     globalsolution)
    # final_state = simtracker.iter == 1 ?  
    #                         solution2state(
    #                             track.S[end], 
    #                             globalsolution) : 
    #                         simtracker.prevsol[end]



    # fit model
    _, _, control_model, solution = run_mtm(
        :dynamics,
        3;
        track=track,
        icond=initial_state,
        # fcond=final_state,
        control_options=:default,
        showplots=false,
        n_iter=5000,
        quiet=true,
    )


    if "LOCALLY_SOLVED" != string(termination_status(control_model))
        @warn "Could not solve $(simtracker.iter)" termination_status(control_model)
        # success = false
        success = true
    else
        success = true
    end

    simtracker.prevsol = solution
    return initial_state, solution, track, success
end


function run_simulation(; n_iter = 200, Δt=.05)
    # run global solution
    _, bike, _, globalsolution = run_mtm(
        :dynamics,
        3;
        control_options=:default,
        showplots=false,
        n_iter=5000,
    )

    # plot background & global trajectory
    p1 = draw(:arena)
    draw!(FULLTRACK; alpha=.1)
    plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_grey_dark, lw=6, alpha=.5, label=nothing)


    # simulate iteratively
    colors = [
            range(HSL(colorant"red"), stop=HSL(colorant"green"), length=(Int ∘ floor)(n_iter/2))...,
            range(HSL(colorant"green"), stop=HSL(colorant"blue"), length=(Int ∘ ceil)(n_iter/2))...
    ]
    simtracker = SimTracker(Δt=Δt)
    for i in pbar(1:n_iter, redirectstdout=false)
        simtracker.iter = i
        initial_state, solution, track, success = step(simtracker, globalsolution)
        

        # plot stuff
        # p1 = draw(:arena)
        # draw!(FULLTRACK; alpha=.1)
        # plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_grey_dark, lw=4, label=nothing)
    
        # draw!(track; alpha=1, color="black")
        plot_bike_trajectory!(solution, bike; showbike=false, label=nothing, color=colors[i], alpha=.8, lw=4)

        # draw initial and final states
        draw!(initial_state; color=colors[i], alpha=1)
        # draw!(final_state; color=blue_dark)
        plot!(; title="Iter $i, time: $(round(simtracker.Δt * i; digits=2))s")
        # sleep(2)
        display(p1)

        success || break
        simtracker.s > 250 && break
    end


    # show all plots
    display(p1)




    nothing
end


# TODO when the simulation includes the end of the track we should set end conditions.

run_simulation()