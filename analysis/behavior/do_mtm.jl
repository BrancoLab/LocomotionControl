"""
Run MTM problem and plot results
"""

using Term
import Term.consoles: clear
install_stacktrace()
install_term_logger()

import jcontrol: run_mtm

# clear console
# clear()

print("\n\n" * hLine("start"; style="bold green"))
track, bike, control_model, solution = run_mtm(
    :kinematics,  # model type
    200;  # number of supports
    realistic_controls=false,
    showtrials=nothing,
    niters=5000,
    timed=true,
    showplots=true,
)

print("\n", hLine("done"; style="bold blue") * "\n\n")
