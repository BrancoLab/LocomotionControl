using Plots
using MyterialColors: red, black, indigo, grey_dark, salmon_dark, white
import InfiniteOpt: InfiniteModel, value



"""
Plot summary of MTM solution
"""
function summary_plot(model::InfiniteModel, wrt::Symbol)
    xvals = wrt == :space ? value(model[:s]) : value(model[:t])
    # s = value(model[:s])

    p(x, color::String) = plot(xvals, value(model[x]), lw=4, label=string(x), color=color)
    p(x, transform::Function) = plot(xvals, transform.(value(model[x])), lw=4, label=string(x), color=black)
    p(x) = p(x, black)

    display(
        plot(
            p(:n),
            p(:ψ),
            p(:v),
            p(:δ, rad2deg),
            p(:β, rad2deg),
            p(:ω, rad2deg),
            p(:SF),
            plot(),
            p(:uv, red),
            p(:uδ, red),
            layout=grid(5, 2),
            size=(1200, 1200)
        )
    )

end


# """
# Plot the bike's posture every n frames
# """
function plot_bike(_plot, model::Solution, bike::Bicycle, n::Int)

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

    function plot_wheel(x, y, lw, color)
        plot!(_plot, x, y, lw=lw, color=color, label=nothing)
        scatter!(_plot, x, y, ms=lw/2, color=color, label=nothing, msw=0)

    end
    
    _L = bike.L/2
    for t in 1:n:length(model.x)
        x, y, θ, δ = model.x[t], model.y[t], model.θ[t], model.δ[t]
        
        # plot bike's frame
        bx, by = points(x, y, θ, _L)
        plot!(_plot, bx, by, lw=bike.body_lw, color=bike.body_color, label=nothing)
        scatter!(_plot, [x], [y], ms=8, mc=white, msc=salmon_dark, label=nothing)

        # plot wheels (first white background)
        plot_wheel(points(bx[1], by[1], θ+δ, bike.wheel_length)..., bike.wheel_lw, bike.front_wheel_color)
        plot_wheel(points(bx[2], by[2], θ, bike.wheel_length)..., bike.wheel_lw, bike.wheel_color)
    end
end



function summary_plot(model::Solution, controlmodel::InfiniteModel,  track::Track, bike::Bicycle)

    # plot the track + XY trajectory
    xyplot = plot(
        track.X, track.Y, 
        lw=5,
        lc=grey_dark,
        label=nothing,
        xlabel="X",
        ylabel="Y",
        ls=:dash,
        aspect_ratio=:equal,
        title="Duration: $(round(model.t[end]; digits=3))s",
        size=(1200, 1200),
        alpha=.6
    )

    for border in get_track_borders(track)
        plot!(
            xyplot,
            border.X, border.Y, 
            lw=3,
            lc=grey_dark,
            label=nothing,
            alpha=.6
        )
    end

    # mark bike's trajectory
    plot!(xyplot, model.x, model.y, color=salmon_dark, lw=6, label="model")
    n = (Int ∘ round)(1 / model.δt * .5)  # put a mark ever .25s aprox 
    scatter!(xyplot, model.x[1:n:end], model.y[1:n:end], ms=6, color=black, label=nothing)

    # plot bike's posture
    plot_bike(xyplot, model, bike, n)

    t = model.t
    _t = value(controlmodel[:t])

    function plot_two(x1, x2, y1, y2, n1, n2)
        p = plot(x1, y1, label=n1, color=black, lw=6)
        plot!(p, x2, y2, label=n2, color=red, lw=4)
        return p
    end


    display(
        plot(
            xyplot,
            plot(t, rad2deg.(model.θ), label="θ", ;w=2, color=black),
            # plot(t, model.v, label="v", ;w=2, color=black),
            # plot(t, rad2deg.(model.δ), label="δ", ;w=2, color=black),
            plot_two(t, _t, model.v, value(controlmodel[:v]), "ODE v", "control v"),
            plot_two(t, _t, rad2deg.(model.δ), rad2deg.(value(controlmodel[:δ])), "ODE δ", "control δ"),

            # plot(t, model.uv, label="uv", ;w=2, color=red),
            plot_two(t, _t, model.uv, value(controlmodel[:uv]), "ODE uv", "control uv"),
            plot_two(t, _t, model.uδ, value(controlmodel[:uδ]), "ODE uδ", "control uδ"),
            # plot(t, model.uδ, label="uδ", ;w=2, color=red),
            layout=grid(3, 2),
            size=(1200, 1200)
        )
    )


    display(xyplot)
end