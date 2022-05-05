import jcontrol: run_mtm, toDict, State

using CSV, DataFrames



_, _, _, globalsolution = run_mtm(
    :dynamics,
    3;
    showtrials=nothing,
    n_iter=5000,
    fcond=State(; u=30, ω=0),
    timed=false,
    showplots=false,
)


fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior"
destination = joinpath(fld, "globalsolution.csv")
data = DataFrame(toDict(globalsolution))
CSV.write(destination, data)