"""
Run MTM problem and plot results
"""

using Term
# install_stacktrace()
install_term_logger()

import jcontrol.control: ControlOptions, Bounds
import jcontrol.bicycle: State
import jcontrol: run_mtm, Track

print("\n\n" * hLine("start"; style="bold green"))


# --------------------------------- kinematic -------------------------------- #
# track, bike, control_model, solution = run_mtm(
#     :kinematics,  # model type
#     3;  # supports density
#     control_options=:default,
#     showtrials=50,
#     n_iter=5000,
#     timed=false,
#     showplots=true,
# )


# ---------------------------------- dynamic---------------------------------- #

track = Track(;start_waypoint=2, keep_n_waypoints=-1)

coptions = ControlOptions(;
u_bounds=Bounds(10, 80),
δ_bounds=Bounds(-120, 120, :angle),
δ̇_bounds=Bounds(-5, 5),
ω_bounds=Bounds(-600, 600, :angle),
v_bounds=Bounds(-250, 250),
Fu_bounds=Bounds(-250, 3000),
)

icond = State(; u=10)
fcond = State(; u=30, ω=0)

track, bike, control_model, solution = run_mtm(
    :dynamics,  # model type
    3;  # supports density
    track=track,
    control_options=coptions,
    icond=icond,
    fcond=fcond,
    showtrials=25,
    n_iter=5000,
    timed=false,
    showplots=true,
)

print("\n", hLine("done"; style="bold blue") * "\n\n")
