import jcontrol: run_mtm, toDict, State
import jcontrol.control: ControlOptions, Bounds
import jcontrol.bicycle: Bicycle
using CSV, DataFrames

_, _, _, globalsolution = run_mtm(
    :dynamics, 2; showtrials=nothing, n_iter=5000, timed=false, showplots=true
)

# fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior"
fld = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\behavior"
destination = joinpath(fld, "globalsolution.csv")
data = DataFrame(toDict(globalsolution))
CSV.write(destination, data)
