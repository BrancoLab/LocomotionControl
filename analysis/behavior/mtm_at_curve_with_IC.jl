# load basic packages
include("./analysis_fixtures.jl")

import Term.progress: @track
import Term: Panel, highlight, theme
import InfiniteOpt: termination_status


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
    1=>0, 2=> 0, 3=> 0, 4=> 0
)
globalsolution = load_global_solution()


# CO = ControlOptions(;
# u_bounds=Bounds(10, 100),
# δ_bounds=Bounds(-90, 90, :angle),
# δ̇_bounds=Bounds(-8, 8),
# ω_bounds=Bounds(-600, 600, :angle),
# v_bounds=Bounds(-25, 25),
# Fu_bounds=Bounds(-4500, 4500),
# )

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
    v_angle = atan.(Δ(movingaverage(trial.y, 6)) .* 60, Δ(movingaverage(trial.x, 6)) .* 60)

    # get β
    β = (unwrap(v_angle) .- movingaverage(trial.θ, 6)) .+ 2π

    # get v
    v = movingaverage(tan.(β) .* movingaverage(trial.u, 6), 6)

    # get δ 
    c, l_r, l_f, m = bike.c, bike.l_r, bike.l_f, bike.m
    δ = atan.((l_f+l_r)/l_r * tan.(β))
    δ̇ = Δ(δ) .* 60

    # get lateral forces
    u = movingaverage(trial.u, 6)
    ω = movingaverage(trial.ω, 6)
    u̇ = Δ(u) .* 60
    # v̇ = Δ(v)
    # ω̇ = Δ(ω)

    Ff = @. c*(δ - (l_f*ω + v)/u)
    # Fr = @. c*(l_r * ω - v)/u

    # Get Fu
    Fu = @. m*u̇ + Ff*sin(δ) - m*ω*v

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
        δ=δ[frame],
        δ̇=0.0,
        Fu=Fu[frame],
        β= _β,
        v= -v[frame],
        smoothing_window=6,
    )

    # get δ
    # initial_state.δ = sign(initial_state.ψ) == sign(sstate.ψ) ? sstate.δ : - sstate.δ
    # initial_state.δ̇ = sign(initial_state.ψ) == sign(sstate.ψ) ? sstate.δ̇ : - sstate.δ̇

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


    initial_state.ψ /= 4
    # initial_state.u = 10


    return sstate, initial_state
end


# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

bigbike = Bicycle(; width=1.0)
smallbike = Bicycle(width=1.0)

# iterate over curves
for (i, curve) in enumerate(curves)
    @info "processing curve $i"
    i == 1  && continue

    # iterate over trials
    successess = 0
    @track for (n, trial) in enumerate(trials)
        # n != 25 && continue
        n % 5 != 0 && continue
        n == 200 && break

        s_start = curve.s0 + Δs[i]
        _, _, start, stop = get_curve_indices(trial, s_start, curve.sf)

        # get initial conditions
        gs_initial_state, initial_state = initial_conditions(trial, start, s_start)
        bike = abs(initial_state.n) > 1.2 ? smallbike : bigbike
        
        # get track and final state
        track_len = curve.sf - s_start + 20
        track = jc.trim(FULLTRACK, s_start - 10, track_len)

        # run MTM
        _, _, control_model, solution = run_mtm(
            :dynamics,  # model type
            1.0;  # supports density
            showtrials=nothing,
            control_options=:default,
            track=track,
            icond=solution2state(s_start - 10, globalsolution),
            fcond=solution2state(curve.sf + 10, globalsolution),
            showplots=false,
            quiet=true,
            bike=bike,
            waypoint = (s_start, initial_state),
        )

        # inspect solution
        if converged(control_model)
            successess += 1
            @info "SUCCESS"

            destination = joinpath(savepath, "curve_$(i)_trial_$(n).csv")
            data = DataFrame(toDict(solution))
            CSV.write(destination, data)

            ic_tracking = Panel(state2str(initial_state); fit=true, title="tracking", style="red")
            ic_gsol = Panel(state2str(gs_initial_state); fit=true, title="gsol", style="green bold")
            print(ic_tracking * ic_gsol)
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
