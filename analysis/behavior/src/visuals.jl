using Plots
using MyterialColors: red, black, indigo, grey_dark, salmon_dark, white, grey
import InfiniteOpt: InfiniteModel, value
import DataFrames: DataFrame


plot_arena!() = plot!([-8, 47], [-12, 66], reverse(arena, dims = 1), yflip = false)
plot_arena() = plot([-8, 47], [-12, 66], reverse(arena, dims = 1), yflip = false)



"""
Plot summary of MTM solution
"""
function summary_plot(model::InfiniteModel, wrt::Symbol)
    xvals = wrt == :space ? value(model[:s]) : value(model[:t])
    # s = value(model[:s])

    fig = plot(; layout=grid(5, 2), size=(1200, 1200))

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
    p(3, :v)
    p(4, :δ, rad2deg)
    p(5, :β, rad2deg)
    p(6, :ω, rad2deg)
    p(7, :SF)
    plot!(; subplot=8)
    p(9, :uv, red)
    p(10, :uδ, red)
    display(fig)
    return nothing
end


"""
Plots all tracking data form a dataframe of trials
"""
function plot_trials!(trials::DataFrame)
    for (i, trial) in enumerate(eachrow(trials))
        plot!(trial.body_x, trial.body_y, color=grey, lw=1.5, label=nothing)
    end
end


# """
# Plot the bike's posture every n frames
# """
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

    _L = bike.L / 2
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

function summary_plot(
    model::Solution, controlmodel::InfiniteModel, track::Track, bike::Bicycle; trials::Union{Nothing, DataFrame}=nothing
)
    # plot the track + XY trajectory
    xyplot = plot_arena()
    plot!(
        track.X,
        track.Y;
        lw=5,
        lc=black,
        label=nothing,
        xlabel="X",
        ylabel="Y",
        ls=:dash,
        aspect_ratio=:equal,
        title="Duration: $(round(model.t[end]; digits=3))s",
        size=(1200, 1200),
        alpha=0.8,
    )

    for border in get_track_borders(track)
        plot!(border.X, border.Y; lw=4, lc=black, label=nothing, alpha=1)
    end

    # plot trials
    if !isnothing(trials)
        plot_trials!(trials)
    end

    # mark bike's trajectory
    plot!(model.x, model.y; color=salmon_dark, lw=6, label="model")
    n = (Int ∘ round)(1 / model.δt * 0.5)  # put a mark ever .25s aprox
    scatter!(model.x[1:n:end], model.y[1:n:end]; ms=6, color=black, label=nothing)

    # plot bike's posture
    plot_bike!(model, bike, n)

    t = model.t
    _t = value(controlmodel[:t])

    function plot_two!(x1, x2, y1, y2, n1, n2; subplot)
        plot!(x1, y1; label=n1, color=black, lw=6, subplot)
        plot!(x2, y2; label=n2, color=red, lw=4, subplot)
        return nothing
    end

    fig = plot(; layout=grid(3, 2), size=(1200, 1200))
    plot!(t, rad2deg.(model.θ), label = "θ", ; w = 2, color = black, subplot=1)
    plot_two!(t, _t, model.v, value(controlmodel[:v]), "ODE v", "control v", subplot=2)
    plot_two!( t, _t, rad2deg.(model.δ), rad2deg.(value(controlmodel[:δ])), "ODE δ", "control δ",subplot=3)
    plot!(t, model.uv, label="uv", ;w=2, color=red, subplot=4)
    plot_two!(t, _t, model.uv, value(controlmodel[:uv]), "ODE uv", "control uv", subplot=5)
    plot_two!(t, _t, model.uδ, value(controlmodel[:uδ]), "ODE uδ", "control uδ", subplot=6)

    display(fig)
    display(xyplot)
    return nothing
end
