using Revise
Revise.revise()

# using Term
# install_term_logger()

using jcontrol

print("\n\n")
print(hLine("start"; style="bold green"))

print(Panel(
"""
Code to run the Kinematic Bicycle model on the full hairpin maze
and solve the minimum time maneuvering problem.

The MTM is solved in the track's curvilinear coordinates system and
then the solution is integrated to get the allocentric coordinates
trajectory over time.
"""
))


# ----------------------------------- params --------------------------------- #
DO_MODEL    = false
DO_FORWWARD = true

δt = 0.01 # Δt for forward integration


# ---------------------------------------------------------------------------- #
#                                DEFINE PARAMERS                               #
# ---------------------------------------------------------------------------- #
# create track object
track = Track(; width=2, keep_n_waypoints=-1, resolution=.001)
@info "track" track

# create bike
bike = Bicycle(L=6, l=2)

coptions = ControlOptions(
    # solver optionsx
    verbose      = 0,
    n_iter       = 200,
    num_supports = 1500,

    # error bounds
    track_safety = .75,
    ψ_bounds     = Bounds(-35, 35, :angle),

    # controls & variables bounds
    uv_bounds = Bounds(-10, 10),          # cm/s²
    uδ_bounds = Bounds(-1.5, 1.5),        # rad/s²

    v_bounds  = Bounds(5, 100),           # cm
    δ_bounds  = Bounds(-80, 80, :angle)   # deg
)

# define initial and final conditions
icond = State(
    x = track.X[1],
    y = track.Y[1],
    θ = track.θ[1],
    v = coptions.v_bounds.lower,
)

fcond = State(
    v = coptions.v_bounds.lower,
)


# ---------------------------------------------------------------------------- #
#                                   FIT MODEL                                  #
# ---------------------------------------------------------------------------- #

if DO_MODEL
    @info "control options" coptions

    control_model = create_and_solve_control(
        track,
        bike,
        coptions,
        icond,
        fcond
    )
end

# visualize results
summary_plot(control_model, :time)

# ---------------------------------------------------------------------------- #
#                              FORWARD INTEGRATION                             #
# ---------------------------------------------------------------------------- #
if DO_FORWWARD
    solution = run_forward_model(
        control_model,
        icond,
        bike;
        δt=δt
    )
end

# visualize
summary_plot(solution, control_model, track, bike)


print("\n")
print(hLine("done"; style="bold blue"))
print("\n\n")
