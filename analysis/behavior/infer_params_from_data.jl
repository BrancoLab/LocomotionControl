using jcontrol
import jcontrol: Δ, unwrap, movingaverage, ∂
using Term
using Plots
import MyterialColors: black, red, blue, indigo, green, teal, salmon, blue_grey
using Statistics: mean, quantile

function summary_statistics(x, xname; lowp=.05, highp=.95)
    _round(x) = round(x; digits=2)
    # @info "x" typeof(x) typeof(x[1]) length(x) lowp highp

    low = quantile(x, highp)

    tprint("""
            [underline green]$xname[/underline green]
        [blue]$highp q:[/blue] [orange]$(_round(low))    [/orange]
        [blue]0.50 q:[/blue] [orange]$(_round(mean(x)))    [/orange]
        [blue]$lowp q:[/blue] [orange]$(_round(quantile(x, lowp)))    [/orange]
    """)
end



function get_params_estimate()
    FPS = 60

    bike = Bicycle()
    L = bike.l_r + bike.l_f


    # load data from file
    data = jcontrol.io.load_trials(; method=:all, keep_n=50)
    
    alldata = Dict(
        "v"=>[],
        "v̇"=>[],
        "δ"=>[],
        "δ̇"=>[]
    )

    fig = plot(; layout=grid(4, 2), size=(1400, 1200))
    for rown in 1:size(data, 1)
        x = data[rown, "body_x"][25:end]
        y = data[rown, "body_y"][25:end]
        xs = data[rown, "snout_x"][25:end]
        ys = data[rown, "snout_y"][25:end]
        xt = data[rown, "tail_base_x"][25:end]
        yt = data[rown, "tail_base_y"][25:end]
        g = data[rown, "gcoord"][25:end]

        # compute velocity
        v = movingaverage(
            sqrt.(Δ(x) .^ 2 .+ Δ(y) .^ 2) * FPS,
            12
            )
        
        # get data only when v is high enough and close to start
        cut = findfirst(v .> 25)
        g[cut] > .1 && continue
        v = v[cut:end]


        # get acceleration as derivative

        time = (collect(0:length(v)) ./ FPS)[2:end]
        a = ∂(time, v)

        # get orientation and angular velocity
        θ = unwrap(atan.(
            ys .- yt , xs .- xt
        ))[cut:end]  # remove some artifacts
        θ = rad2deg.(θ)
        # θ .-= θ[200]
        θ = movingaverage(θ, 6)

        ω = ∂(time, θ)


        # compute δ and δ̇
        δ = atan.(
            (L.* deg2rad.(ω)), v 
        )
        δ = movingaverage(δ, 6)
        δ = rad2deg.(δ)
        δ̇ = ∂(time, δ)

        # append data
        append!(alldata["v"], v)
        append!(alldata["v̇"], a)
        append!(alldata["δ"], δ)
        append!(alldata["δ̇"], δ̇)
        # append!(alldata["ω"], ω)


        # plot quantities
        g = g[cut:end]
        plot!(g, x[cut:end], subplot=1, label=nothing, alpha=.5, lw=3, color=black)
        plot!(g, y[cut:end], subplot=2, label=nothing, alpha=.5, lw=3, color=red)
        plot!(g, v, subplot=3, label=nothing, alpha=.5, lw=3, color=blue)
        plot!(g, θ, subplot=4, label=nothing, alpha=.5, lw=3, color=green)
        plot!(g, a, subplot=5, label=nothing, alpha=.5, lw=3, color=indigo)
        plot!(g, ω, subplot=6, label=nothing, alpha=.5, lw=3, color=teal)


        plot!(g, δ, subplot=7, label=nothing, alpha=.5, lw=3, color=salmon)
        plot!(g, δ̇, subplot=8, label=nothing, alpha=.5, lw=3, color=blue_grey)

    end

    # style axes
    plot!(subplot=1, ylabel="x position (cm)")
    plot!(subplot=2, ylabel="y position (cm)")
    plot!(subplot=3, ylabel="v (cm/s)")
    plot!(subplot=4, ylabel="θ (deg)")
    plot!(subplot=5, ylabel="v̇ (cm/s²)")
    plot!(subplot=6, ylabel="ω (deg/sec)")
    plot!(subplot=7, ylabel="δ (deg)")
    plot!(subplot=8, ylabel="δdot(deg/s)")

    display(fig)


    # print summary statistics
    for (k,v) in zip(keys(alldata), values(alldata))
        # println(k, size(v))
        summary_statistics(v, k)
    end

end

print("\n"^5)
get_params_estimate()