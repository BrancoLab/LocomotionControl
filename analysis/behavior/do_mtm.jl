"""
Run MTM problem and plot results
"""

using Term
# install_stacktrace()
install_term_logger()

import jcontrol: run_mtm


print("\n\n" * hLine("start"; style="bold green"))

track, bike, control_model, solution = run_mtm(
    :kinematics,  # model type
    3;  # supports density
    control_options=:default,
    showtrials=nothing,
    n_iter=5000,
    timed=false,
    showplots=true,
)


# track, bike, control_model, solution = run_mtm(
#     :dynamics,  # model type
#     3;  # supports density
#     control_options=:realistic,
#     showtrials=50,
#     n_iter=5000,
#     timed=true,
#     showplots=true,
# )

print("\n", hLine("done"; style="bold blue") * "\n\n")
