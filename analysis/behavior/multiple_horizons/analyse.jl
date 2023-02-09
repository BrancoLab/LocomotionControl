using Plots, CSV
import DataFrames: DataFrame
using Glob
import MyterialColors: blue_grey_dark, blue_grey
import Colors: HSL
using NaturalSort

import jcontrol: Bicycle, FULLTRACK, naturalsort, Solution
import jcontrol.io: PATHS, load_cached_trials
using jcontrol.visuals
import jcontrol: trimtrial
import jcontrol.forwardmodel: trimsolution

S0 = 0
S1 = 260

trials = map(t -> trimtrial(t, S0, S1), load_cached_trials(; keep_n=nothing))

bike = Bicycle()
globalsolution = DataFrame(
    CSV.File(joinpath(PATHS["horizons_sims_cache"], "global_solution.csv"))
)

globalsolution = trimsolution(Solution(globalsolution), S0, S1)

# initialize plots with global solution
xyplot = draw(:arena; legend=:bottomleft)
plot_bike_trajectory!(
    globalsolution,
    bike;
    showbike=false,
    color=blue_grey_dark,
    lw=6,
    alpha=0.8,
    label="global solution",
)

uplot = plot(; xlim=[0, 200], ylim=[0, 100], ylabel="long.speed. (cm/s)")
progressplot = plot(; ylabel="time (s)", xlabel="s (cm)", legend=:bottomright)

# plot trials kinematics
for trial in trials
    plot!(uplot, trial.s, trial.u; color="black", alpha=0.015, label=nothing)
end

# plot global solution kinematics
global_kwargs = Dict(:label => nothing, :color => blue_grey_dark, :lw => 6, :alpha => 0.4)
plot!(uplot, globalsolution.s, globalsolution.u; global_kwargs...)
plot!(progressplot, globalsolution.s, globalsolution.t; global_kwargs...)

# durations histogram
h = histogram(
    map(t -> t.duration, trials);
    label="trial durations",
    color="black",
    alpha=0.2,
    xlim=[0, 15],
)
plot!(
    h,
    [globalsolution.t[end], globalsolution.t[end]],
    [0, 150];
    color=blue_grey_dark,
    lw=6,
    alpha=1,
    label="global solution",
)

# plot simulations
files = sort(
    glob("multiple_horizons_mtm_horizon_length*.csv", PATHS["horizons_sims_cache"]);
    lt=natural,
)
println.(files)
colors = range(HSL(326, 0.9, 0.68); stop=HSL(212, 0.9, 0.6), length=max(2, length(files)))
alphas = range(0.9, 0.6; length=max(5, length(files)))
for (file, color, alpha) in zip(files, colors, alphas)
    data = trimsolution(Solution(DataFrame(CSV.File(file))), S0, S1)
    name = "horizon length: " * (split(file, "_")[end][1:(end - 4)])

    plot_kwargs = Dict(:lw => 4, :label => name, :color => color, :alpha => 1)

    # plot trajectory
    plot!(xyplot, data.x, data.y; lw=4, color=color, label=name)

    # plot kinematics
    plot!(uplot, data.s, data.u; plot_kwargs...)
    scatter!(uplot, [data.s[end]], [data.u[end]]; markersize=10, color=color, label=nothing)
    plot!(progressplot, data.s, data.t; plot_kwargs...)
    plot!(h, [data.t[end], data.t[end]], [0, 100]; plot_kwargs...)
end

# layout and plot
l = @layout [
    b{0.4h}
    c{0.4h}
    d{0.2h}
]

# display(plot(xyplot, size=(800, 1200) ))
display(plot(uplot, progressplot, h; layout=l, size=(1400, 1200)))
