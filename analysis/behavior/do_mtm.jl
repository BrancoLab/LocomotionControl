"""
Run MTM problem and plot results
"""

using Term
# install_stacktrace()
install_term_logger()

import jcontrol: run_mtm


print("\n\n" * hLine("start"; style="bold green"))

# track, bike, control_model, solution = run_mtm(
#     :kinematics,  # model type
#     3;  # supports density
#     control_options=:default,
#     showtrials=50,
#     n_iter=5000,
#     timed=false,
#     showplots=true,
# )

import jcontrol.control: ControlOptions, Bounds
import jcontrol.bicycle: State

coptions = ControlOptions(
    u_bounds = Bounds(5, 80),
    δ_bounds = Bounds(-50, 50, :angle),
    δ̇_bounds = Bounds(-4, 4),
    ω_bounds = Bounds(-400, 400, :angle),

    Fy_bounds = Bounds(-250, 250),
    v_bounds = Bounds(-125, 125),
    Fu_bounds = Bounds(-250, 250)
)

icond = State(
    u = 10,
)

track, bike, control_model, solution = run_mtm(
    :dynamics,  # model type
    3;  # supports density
    control_options=coptions,
    icond = icond,
    showtrials=50,
    n_iter=5000,
    timed=false,
    showplots=true,
)

print("\n", hLine("done"; style="bold blue") * "\n\n")
