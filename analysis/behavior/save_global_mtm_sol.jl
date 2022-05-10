import jcontrol: run_mtm, toDict, State
import jcontrol.control: ControlOptions, Bounds
import jcontrol.bicycle: Bicycle
using CSV, DataFrames


# bike = Bike()
CO = ControlOptions(
    u_bounds=Bounds(10, 80),
    δ_bounds=Bounds(-30, 30, :angle),
    δ̇_bounds=Bounds(-10, 10),
    ω_bounds=Bounds(-600, 600, :angle),
    v_bounds=Bounds(-6, 6),
    Fu_bounds=Bounds(-4500, 4500),
)

bike = Bicycle(;
    m_f=15,
    m_r=18,
    c=6e3
)


_, _, _, globalsolution = run_mtm(
    :dynamics,
    2;
    showtrials=nothing,
    bike=bike,
    control_options=CO,
    n_iter=5000,
    fcond=State(; u=30, ω=0),
    timed=false,
    showplots=false,
)


fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior"
destination = joinpath(fld, "globalsolution_bad_params4.csv")
data = DataFrame(toDict(globalsolution))
CSV.write(destination, data)