using Plots
import MyterialColors: black
using Term
import Term: track as pbar
install_term_logger()
# install_stacktrace()

using jcontrol
using jcontrol.visuals
import jcontrol: closest_point_idx, euclidean

function compare()
    # ---------------------------------- run MTM --------------------------------- #

    track, bike, _, solution = run_mtm(
        :kinematics,  # model type
        3;  # supports density
        control_options=:default,
        showtrials=nothing,
        n_iter=5000,
        timed=false,
        showplots=false,
    )

    # plot model trajectory
    plt = plot_arena()
    plot_track!(track)
    plot_bike_trajectory!(solution, bike; showbike=false)

    # -------------------------- do comparison with data ------------------------- #
    # load data
    cpoints = get_comparison_points(track; δs=5)
    trials = load_trials(; keep_n=20)

    # show data
    plot_trials!(trials; lw=3)
    plot_comparison_point!.(cpoints.points)

    # do comparisons
    Δd::Vector{Float64} = []
    Δθ::Vector{Float64} = []
    Δu::Vector{Float64} = []
    Δω::Vector{Float64} = []

    uplot = plot(; title="speed", xlabel="s (cm)", ylabel="speed cm/s", legend=false)
    ωplot = plot(; title="ang.vel.", xlabel="s (cm)", ylabel="avel rad/s", legend=false)

    for trial in pbar(
        eachrow(trials); description="Iterating trials", expand=true, columns=:detailed
    )
        trial = Trial(trial, track)

        plot!(uplot, trial.s, trial.u; color=black, alpha=0.7)
        plot!(ωplot, trial.s, trial.ω; color=black, alpha=0.7)

        for cp in cpoints.points
            cp.s < trial.s[1] && continue

            # get closest trial point
            idx = closest_point_idx(trial.x, cp.x, trial.y, cp.y)
            x, y, θ = trial.x[idx], trial.y[idx], trial.θ[idx]
            # mark point for debugging
            scatter!(plt, [x], [y]; ms=5, color="blue", label=nothing, alpha=0.5)

            # get the closest model point
            idxs = closest_point_idx(solution.x, cp.x, solution.y, cp.y)

            # get errors
            push!(Δd, euclidean(solution.x[idxs], x, solution.y[idxs], y))
            push!(Δθ, mod(solution.θ[idxs] - θ, 2π))
            push!(Δu, solution.u[idxs] - trial.u[idx])
            push!(Δω, solution.ω[idxs] - trial.ω[idx])

            scatter!(uplot, [cp.s], [solution.u[idxs]]; color="red", ms=10)
            scatter!(ωplot, [cp.s], [solution.ω[idxs]]; color="red", ms=10)

            # break
        end
        # break
    end

    # ------------------------------------ fin ----------------------------------- #

    l = @layout [
        a{0.5w} grid(4, 1)
        b{0.15h}
        c{0.15h}
    ]
    return display(
        plot(
            plt,
            histogram(Δd; label="Δd"),
            histogram(Δu; label="Δu"),
            histogram(Δω; label="Δω"),
            histogram(Δθ; label="Δθ"),
            uplot,
            ωplot;
            layout=l,
            size=(1600, 1200),
        ),
    )
end

compare()

# TODO trim trials based on when speed is high enough
# TODO remove artifacts from angular velocity
