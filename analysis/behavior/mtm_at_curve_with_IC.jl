# load basic packages
include("./analysis_fixtures.jl")

import Term.progress: @track
import Term: Panel, highlight, theme
import InfiniteOpt: termination_status


import jcontrol.Run: run_mtm
import jcontrol.bicycle: State
import jcontrol: toDict
import jcontrol as jc
import jcontrol.visuals: draw
import jcontrol.control: default_control_options
import jcontrol.forwardmodel: solution2state

savepath = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/mtm_on_curve"

state2str(state) = highlight(string(state); theme=theme)
converged(control_model) = "LOCALLY_SOLVED" == string(termination_status(control_model))


Δs = Dict(
    1=>0, 2=>5, 3=>0, 4=>5
)

# run MTM on the whole track
# _, _, _, globalsolution = run_mtm(
#     :dynamics,  # model type
#     1,  # supports density
#     showtrials=nothing,
#     fcond=State(; u=20, ψ=0),
#     timed=false,
#     showplots=false,
# )

globalsolution = load_global_solution()

# iterate over curves
for (i, curve) in enumerate(curves)
    @info "processing curve $i"
    i == 1 && continue

    # iterate over trials
    successess = 0
    @track for (n, trial) in enumerate(trials)
        # n != 25 && continue
        # n == 100 && break

        s_start = curve.s0 + Δs[i]
        _, _, start, stop = get_curve_indices(trial, s_start, curve.sf)

        # get initial conditions
        gs_initial_state =  solution2state(s_start, globalsolution)        
        initial_state = State(
            trial, 
            start, 
            FULLTRACK;
            Fu=gs_initial_state.Fu,
            δ=gs_initial_state.δ,
            v=gs_initial_state.v,
            δ̇=gs_initial_state.δ̇,
            β=gs_initial_state.β,
        )
        initial_state.n = -initial_state.n
        initial_state.u = min(
                    max(initial_state.u, default_control_options.u_bounds.lower), 
                    default_control_options.u_bounds.upper
        )
        # initial_state.ψ = gs_initial_state.ψ
        if initial_state.ψ > 6
            initial_state.ψ = 2π - initial_state.ψ
        end
        # initial_state = tr_initial_state
        # initial_state.n = tr_initial_state.n
        # initial_state.ψ = tr_initial_state.ψ
        # initial_state.u = min(80, tr_initial_state.u)

        # print states
        ic_tracking = Panel(state2str(initial_state); fit=true, title="tracking", style="red")
        ic_gsol = Panel(state2str(gs_initial_state); fit=true, title="gsol", style="blue")
        # print(ic_tracking * ic_gsol)
        
        # get track and final state
        track = jc.trim(FULLTRACK, s_start, FULLTRACK.S_f - s_start)
        final_state = solution2state(track.S_f, globalsolution)    

        # run MTM
        _, _, control_model, solution = run_mtm(
            :dynamics,  # model type
            .5;  # supports density
            showtrials=nothing,
            control_options=:default,
            track=track,
            n_iter=5000,
            icond=initial_state,
            fcond=final_state,
            showplots=false,
            quiet=true,
        )

        # inspect solution
        if converged(control_model)
            successess += 1
            @info "SUCCESS"
            print(ic_tracking * ic_gsol)

            destination = joinpath(savepath, "curve_$(i)_trial_$(n).csv")
            data = DataFrame(toDict(solution))
            CSV.write(destination, data)
        else 
            @info "FAILURE"
            print(ic_tracking * ic_gsol)
        end
        sleep(.001)
    end
    @info "Curve $i success rate: $successess / $(length(trials))"
    # break
end
