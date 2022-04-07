module visuals
using Plots
using MyterialColors: red, black, indigo, grey_dark, salmon_dark, white, grey, green_darker
import InfiniteOpt: InfiniteModel, value
import DataFrames: DataFrame
using Images: Images

import jcontrol: Track, get_track_borders, Δ
import ..comparisons: ComparisonPoint, TrackSegment
import ..control: KinematicsProblem, DynamicsProblem
import ..forwardmodel: Solution
import ..bicycle: Bicycle, State
import ..trial: Trial

export draw, draw!, summary_plot
export plot_bike_trajectory!



# ---------------------------------------------------------------------------- #
#                                     ARENA                                    #
# ---------------------------------------------------------------------------- #
arena = Images.load("src/arena.png")


"""
Set figure and ax properties for showing the Track.
"""
function draw!()
    plot!(;
    xlabel="X (cm)",
    ylabel="Y (cm)",
    aspect_ratio=:equal,
    size=(1200, 1200),
    xlim=[-5, 45],
    ylim=[-5, 65],
)
end

""" draw various things by name """
function draw!(what::Symbol; kwargs...)
    if what == :arena
        plt =  plot!(
            [-8, 47],
            [-12, 66],
            reverse(arena; dims=1);
            yflip=false,
            xlim=[-5, 45],
            ylim=[-5, 65],
            kwargs...
        )
        draw!()
        return plt
    end
end


function draw(what::Symbol; kwargs...)
    if what == :arena
        plt =  plot(
            [-8, 47],
            [-12, 66],
            reverse(arena; dims=1);
            yflip=false,
            xlim=[-5, 45],
            ylim=[-5, 65],
            kwargs...
        )
        draw!()
        return plt
    end
end



# ----------------------------------- track ---------------------------------- #
""" Draw track and track borders """
function draw!(track::Track; title="", color=black, lw=5, border_lw=3, alpha=.6, border_alpha=.2)
    plot!(track.X[1:500:end], track.Y[1:500:end]; lw=lw, lc=color, label=nothing, ls=:dash, alpha=alpha)

    for border in get_track_borders(track)
        plot!(border.X[1:500:end], border.Y[1:500:end]; lw=border_lw, lc=color, label=nothing, alpha=border_alpha)
    end
end


# ------------------------------- tracking data ------------------------------ #
"""
Plots all tracking data form a dataframe of trials
"""
function draw!(trials::DataFrame; lw=1.5, color=grey, asscatter=false)
    for (i, trial) in enumerate(eachrow(trials))
        asscatter || plot!(trial.body_x, trial.body_y; color=color, lw=lw, label=nothing)
        asscatter && scatter!(trial.body_x, trial.body_y; color=color, lw=lw, label=nothing)
    end
end

function draw!(trial::Trial; lw=1.5, color=grey, asscatter=false, kwargs...)
    asscatter || plot!(trial.x, trial.y; color=color, lw=lw, label=nothing, kwargs...)
    asscatter && scatter!(trial.x, trial.y; color=color, lw=lw, label=nothing, kwargs...)
end
# -------------------------------- comparisons ------------------------------- #
"""
Draws a line across the track showing the position and 'orientation' of a CP.
"""
function draw!(point::ComparisonPoint; color=black, lw=6, alpha=1)
    dx = point.w * cos.(point.η)
    dy = point.w * sin.(point.η)

    return plot!(
        [point.x - dx, point.x + dx],
        [point.y - dy, point.y + dy];
        lw=lw,
        color=color,
        alpha=alpha,
        label=nothing,
    )
end

"""
    Draw a track segment
"""
function draw!(segment::TrackSegment)
    draw!(
        segment.track; border_lw=6,alpha=1, border_alpha=1.0, color=segment.color
    )

    draw!.(segment.checkpoints; lw=4, alpha=.25)
end

# ----------------------------------- bike ----------------------------------- #
function draw!(state::State; color=black, alpha=1.0)
    plot!(
        [state.x, state.x + 2 * cos(state.θ)],
        [state.y, state.y + 2 * sin(state.θ)],
        lw=6, color="black", label=nothing,
    )

    scatter!([state.x], [state.y], ms=12, color="white", mlc="white", msc="white", lw=0.01, label=nothing, alpha=alpha)

    scatter!([state.x], [state.y], ms=8, color=color, label=nothing, alpha=alpha)
end


# ---------------------------------------------------------------------------- #
#                                 MTM SOLUTION                                 #
# ---------------------------------------------------------------------------- #
function summary_plot(problemtype::KinematicsProblem, model::InfiniteModel, wrt::Symbol)
    xvals = wrt == :space ? value(model[:s]) : value(model[:t])
    # s = value(model[:s])

    fig = plot(; layout=grid(5, 2), size=(1600, 1200))

    function p(i, x, color::String)
        return plot!(xvals, value(model[x]); lw=4, label=string(x), color=color, subplot=i)
    end

    function p(i, x, transform::Function)
        return plot!(
            xvals,
            transform.(value(model[x]));
            subplot=i,
            lw=4,
            label=string(x),
            color=black,
        )
    end

    p(i, x) = p(i, x, black)

    p(1, :n)
    p(2, :ψ)
    p(3, :u)
    p(4, :δ, rad2deg)
    p(5, :β, rad2deg)
    p(6, :ω, rad2deg)
    # p(7, :SF)
    plot!(; subplot=8)
    p(9, :u̇, red)
    p(10, :δ̇, red)
    display(fig)
    return nothing
end

function summary_plot(problemtype::DynamicsProblem, model::InfiniteModel, wrt::Symbol)
    xvals = wrt == :space ? value(model[:s]) : value(model[:t])
    # s = value(model[:s])

    fig = plot(; layout=grid(4, 2), size=(1600, 1200))

    function p(i, x, color::String)
        return plot!(xvals, value(model[x]); lw=4, label=string(x), color=color, subplot=i)
    end

    function p(i, x, transform::Function)
        return plot!(
            xvals,
            transform.(value(model[x]));
            subplot=i,
            lw=4,
            label=string(x),
            color=black,
        )
    end

    p(i, x) = p(i, x, black)

    p(1, :n)
    p(2, :ψ)
    p(3, :u)
    p(4, :δ, rad2deg)
    p(5, :ω, rad2deg)
    # p(6, :SF)
    # p(7, :u̇, red)
    p(8, :δ̇, red)
    display(fig)
    return nothing
end

# ---------------------------------------------------------------------------- #
#                            FORWARD MODEL SOLUTION                            #
# ---------------------------------------------------------------------------- #

"""
Plot the trajectory of the model (forward solution)
showing also the bike's posture
"""
function plot_bike_trajectory!(model, bike; showbike=true, color=salmon_dark, lw=6, alpha=1.0, label="model")
    plot!(model.x, model.y; color=color, lw=lw, label=label, alpha=alpha)

    # plot bike's posture
    return showbike && plot_bike!(model, bike, 50)
end


"""
Simulation summary plot with the bike's trajectory and various
kinematics variables.
"""
function summary_plot(
    model::Solution,
    controlmodel::InfiniteModel,
    track::Track,
    bike::Bicycle;
    trials::Union{Nothing,DataFrame}=nothing,
)
    nsupports = length(value(controlmodel[:u]))
    # plot the track + XY trajectory
    xyplot = draw(:arena)
    draw!(
        track;
        title="Duration: $(round(model.t[end]; digits=3))s | $nsupports supports",
    )

    # plot trials
    if !isnothing(trials)
        draw!(trials)
    end

    # mark bike's trajectory
    plot_bike_trajectory!(model, bike; showbike=true)
    scatter!(
        model.x[1:10:end],
        model.y[1:10:end];
        marker_z=model.u[1:10:end],
        ms=8,
        label=nothing,
    )

    t = model.t
    _t = value(controlmodel[:t])

    function plot_two!(x1, x2, y1, y2, n1, n2; subplot)
        plot!(x1, y1; label=n1, color=black, lw=6, subplot)
        plot!(x2, y2; label=n2, color=red, lw=4, subplot)
        return nothing
    end

    fig = plot(; layout=grid(3, 2), size=(1200, 1200))
    # plot!(t, rad2deg.(model.θ), label = "θ", ; w = 2, color = black, subplot=1)

    plot_two!(t, t, rad2deg.(model.θ), cumsum(model.ω), "model θ", "∫ω"; subplot=1)

    plot_two!(
        t,
        _t,
        rad2deg.(model.δ),
        rad2deg.(value(controlmodel[:δ])),
        "ODE δ",
        "control δ";
        subplot=2,
    )
    plot_two!(t, _t, model.u, value(controlmodel[:u]), "ODE u", "control u"; subplot=3)
    plot_two!(
        t,
        _t,
        rad2deg.(model.ω),
        rad2deg.(value(controlmodel[:ω])),
        "ODE ω",
        "control ω";
        subplot=4,
    )
    plot_two!(
        t,
        _t,
        rad2deg.(model.δ̇),
        rad2deg.(value(controlmodel[:δ̇])),
        "ODE δ̇dot",
        "control δ̇dot";
        subplot=6,
    )
    plot_two!(t, _t, model.Fu, value(controlmodel[:Fu]), "ODE u̇", "control u̇", subplot=5)

    display(fig)
    display(xyplot)
    return nothing
end



# ---------------------------------------------------------------------------- #
#                                     BIKE                                     #
# ---------------------------------------------------------------------------- #
"""
Plot the bike's posture every n frames
"""
function plot_bike!(model::Solution, bike::Bicycle, n::Int)
    """
    Get points to plot a line centered at (x,y) with angle α and length l
    """
    function points(x, y, α, l)
        x_front = l * cos(α) + x
        y_front = l * sin(α) + y

        x_back = l * cos(α + π) + x
        y_back = l * sin(α + π) + y
        return [x_front, x_back], [y_front, y_back]
    end

    function plot_wheel!(x, y, lw, color)
        plot!(x, y; lw=lw, color=color, label=nothing)
        scatter!(x, y; ms=lw / 2, color=color, label=nothing, msw=0)
        return nothing
    end

    _L = (bike.l_f + bike.l_r) / 2
    for t in 1:n:length(model.x)
        x, y, θ, δ = model.x[t], model.y[t], model.θ[t], model.δ[t]

        # plot bike's frame
        bx, by = points(x, y, θ, _L)
        plot!(bx, by; lw=bike.body_lw, color=bike.body_color, label=nothing)
        scatter!([x], [y]; ms=8, mc=white, msc=salmon_dark, label=nothing)

        # plot wheels (first white background)
        plot_wheel!(
            points(bx[1], by[1], θ + δ, bike.wheel_length)...,
            bike.wheel_lw,
            bike.front_wheel_color,
        )
        plot_wheel!(
            points(bx[2], by[2], θ, bike.wheel_length)..., bike.wheel_lw, bike.wheel_color
        )
    end
end



end
