# load basic packages
include("./analysis_fixtures.jl")

import Term.progress: @track
import Term: Panel, highlight, theme
import InfiniteOpt: termination_status
using Random, Distributions


import jcontrol.Run: run_mtm
import jcontrol.bicycle: State, Bicycle
import jcontrol: toDict, unwrap, movingaverage, Δ
import jcontrol as jc
import jcontrol.visuals: draw
import jcontrol.control: default_control_options, ControlOptions, Bounds
import jcontrol.forwardmodel: solution2state


"""
Run MTM with the mice states' as initial conditions for each 
trial at each curve.
"""

# --------------------------------- preamble --------------------------------- #

savepath = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/mtm_on_curve"


Δs = Dict(
    1=>0, 2=> -0, 3=> -0, 4=> -0
)
globalsolution = load_global_solution()




CO = ControlOptions(;
u_bounds=Bounds(10, 100),
δ_bounds=Bounds(-90, 90, :angle),
δ̇_bounds=Bounds(-8, 8),
ω_bounds=Bounds(-600, 600, :angle),
v_bounds=Bounds(-25, 25),
Fu_bounds=Bounds(-4500, 4500),
)

# ----------------------------------- utils ---------------------------------- #

state2str(state) = highlight(string(state); theme=theme)
converged(control_model) = "LOCALLY_SOLVED" == string(termination_status(control_model))

"""
Get initial condition as a state based on tracking data from a trial
at a frame, using inverse kinematics to fill in the missing variables.
Using the global solution to infer variables we can't compute
"""
function initial_conditions(trial, frame, s)
    sstate = solution2state(s, globalsolution)    

    # get the velocity vector angle at each frame
    v_angle = atan.(Δ(movingaverage(trial.y, 6)), Δ(movingaverage(trial.x, 6)))

    # get β
    β = (unwrap(v_angle) .- movingaverage(trial.θ, 6)) .+ 2π

    # get v
    v = movingaverage(tan.(β) .* movingaverage(trial.u, 6), 6)

    # get δ from lateral forecs
    # c, l_r, l_f, m = bike.c, bike.l_r, bike.l_f

    # get Fu
    # u̇ = Δ(trial.u)
    # Fu = m .* u̇ .

    # get values at frame
    _β = β[frame]
    if _β > 6
        _β = _β - 2π
    elseif _β < -6
        _β = _β + 2π
    end

    # get IC
    initial_state = State(
        trial, 
        frame, 
        FULLTRACK;
        Fu=sstate.Fu,
        β= _β,
        v= -v[frame],
        smoothing_window=6,
    )


    # get δ
    initial_state.δ = sign(initial_state.ψ) == sign(sstate.ψ) ? sstate.δ : - sstate.δ
    initial_state.δ̇ = sign(initial_state.ψ) == sign(sstate.ψ) ? sstate.δ̇ : - sstate.δ̇

    # add noise to initial state
    # noise_state = solution2state(s, globalsolution)    
    # noise_state.n += rand(Normal(0, 0.5))
    # noise_state.ψ += rand(Normal(0, 0.5))
    # noise_state.u += rand(Normal(0, 10))
    # noise_state.v += rand(Normal(0, 2.5))
    # noise_state.ω += rand(Normal(0, 3.0))
    # noise_state.δ += rand(Normal(0, 2.0))
    # noise_state.δ̇ += rand(Normal(0, 2.5))
    # noise_state.Fu += rand(Normal(0, 1000))



    # enforce variable bounds
    _vars = Dict(
        :u_bounds => :u,
        :ψ_bounds => :ψ,
        :δ̇_bounds => :δ̇,
        :ω_bounds => :ω,
        :Fu_bounds => :Fu,
        :v_bounds => :v,
    )

    for (b,v) in pairs(_vars)
        bounds = getfield(default_control_options, b)
        setfield!(
            initial_state, 
            v, 
            clamp(
                getfield(initial_state, v), 
                bounds.lower, bounds.upper
            )
        )
    end


    return sstate, initial_state
end


# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

bigbike = Bicycle()
smallbike = Bicycle(width=1.0)

# iterate over curves
for (i, curve) in enumerate(curves)
    @info "processing curve $i"
    i != 2  && continue

    # iterate over trials
    successess = 0
    @track for (n, trial) in enumerate(trials)
        # n != 25 && continue
        # n == 100 && break

        s_start = curve.s0 + Δs[i]
        _, _, start, stop = get_curve_indices(trial, s_start, curve.sf)

        # get initial conditions
        gs_initial_state, initial_state = initial_conditions(trial, start, s_start)
        bike = abs(initial_state.n) > 1.2 ? smallbike : bigbike
        # bike = smallbike
        
        # get track and final state
        track = jc.trim(FULLTRACK, s_start+1, curve.sf - s_start)

        # run MTM
        _, _, control_model, solution = run_mtm(
            :dynamics,  # model type
            1.5;  # supports density
            showtrials=nothing,
            control_options=CO,
            track=track,
            icond=initial_state,
            fcond=solution2state(curve.sf, globalsolution),
            showplots=false,
            quiet=true,
            bike=bike
        )

        # inspect solution
        if converged(control_model)
            successess += 1
            @info "SUCCESS"

            destination = joinpath(savepath, "curve_$(i)_trial_$(n).csv")
            data = DataFrame(toDict(solution))
            CSV.write(destination, data)
        else 
            @info "FAILURE" termination_status(control_model)

            ic_tracking = Panel(state2str(initial_state); fit=true, title="tracking", style="red")
            ic_gsol = Panel(state2str(gs_initial_state); fit=true, title="gsol", style="blue")
            print(ic_tracking * ic_gsol)
        end
        sleep(.001)
    end
    @info "Curve $i success rate: $successess / $(length(trials))"
end
