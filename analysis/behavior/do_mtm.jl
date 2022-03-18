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
#     300;  # number of supports
#     realistic_controls=false,
#     showtrials=50,
#     niters=5000,
#     timed=false,
#     showplots=true,
# )


track, bike, control_model, solution = run_mtm(
    :dynamics,  # model type
    200;  # number of supports
    realistic_controls=true,
    showtrials=50,
    niters=5000,
    timed=true,
    showplots=true,
)

print("\n", hLine("done"; style="bold blue") * "\n\n")

         