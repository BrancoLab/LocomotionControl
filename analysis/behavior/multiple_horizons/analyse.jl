using Plots, CSV
import DataFrames: DataFrame
using Glob
import MyterialColors: blue_dark

import jcontrol: Bicycle
import jcontrol.io: PATHS, load_cached_trials
using jcontrol.visuals


trials = load_cached_trials(; keep_n = 20,)

bike = Bicycle()
globalsolution = DataFrame(CSV.File(joinpath(PATHS["horizons_sims_cache"], "global_solution.csv")))

for file in glob("multiple_horizons_mtm_horizon_length*.csv", PATHS["horizons_sims_cache"])
    print(file)
    data = DataFrame(CSV.File(file))
    print()


    p1 = draw(:arena)
    draw!(FULLTRACK; alpha=.1)
    draw!.(trials; lw=3)    



    plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_dark, lw=6, alpha=.8, label="global")
    plot!(data.x, data.y, lw=5, color="red", label="Short")

    p2 = plot(globalsolution.s, globalsolution.u, label="global", color=blue_dark)
    plot!(data.s, data.u, label="short", color="red")

    display(plot(
            p1, p2
            
            )
            )

    break
end


