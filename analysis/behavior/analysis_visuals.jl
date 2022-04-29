"""
Collection of plotting functions, used mostly in Thesis.
"""


using Plots
using KernelDensity
using StatsBase


"""
Plot the 95th confidence interval of X as a vertical bar at the x axis value xpos
"""
function plot_CI!(X, xpos, color; kwargs...)
    low, med, high = percentile(X, [2.5, 50, 97.5])

    plot!(
        [xpos, xpos], [low, high],
        color=color,
        lw=12;
        kwargs...
    )
    Δ = max(0.02 * med, .2)
    plot!(
        [xpos, xpos], [med - Δ, med + Δ],
        color="white",
        lw=10, label=nothing
    )
end


"""
Plot the KDE of X as a density plot with the 95th CI underneath a a bar
"""
function plot_kde_and_CI(X; bandwidth=1.0, xlabel="", color="black", ylabel="", label=nothing)
    B = kde(Vector{Float64}(X); bandwidth=bandwidth)
    low, med, high = percentile(X, [2.5, 50, 97.5])

    plt = plot(
        B.x,
        B.density;
        lw=2,
        color=color,
        fillalpha=.2,
        fillcolor=color,
        fillrange=zeros(length(B.density)),
        xlabel="", ylabel="", label=label
    )

    y = -maximum(B.density) / 20
    plot!(
        [low, high], [y, y], lw=8, color=color, alpha=1, label=nothing
    )
    # plot!([med - med * 0.15, med + med * 0.15], [y, y], lw=4, color="white", alpha=1, label=nothing)
    scatter!([med], [y], color="white", label=nothing, alpha=1, ms=6, shape=:rect)
    return plt
end

function plot_kde_and_CI!(X; bandwidth=1.0, xlabel="", color="black", ylabel="", label=nothing)
    B = kde(Vector{Float64}(X); bandwidth=bandwidth)
    low, med, high = percentile(X, [2.5, 50, 97.5])

    plot!(
        B.x,
        B.density;
        lw=2,
        color=color,
        fillalpha=.2,
        fillcolor=color,
        fillrange=zeros(length(B.density)),
        xlabel="", ylabel="", label=label
    )

    y = -maximum(B.density) / 20
    plot!(
        [low, high], [y, y], lw=8, color=color, alpha=1, label=nothing
    )
    # plot!([med - med * 0.15, med + med * 0.15], [y, y], lw=4, color="white", alpha=1, label=nothing)
    scatter!([med], [y], color="white", label=nothing, alpha=1, ms=6, shape=:rect)
end


"""
More complex 1D KDE plot with options to move and scale the KDE.
"""
function kdeplot!(X;
        scaling=1.0,
        shift=0.0,
        v0=0,
        v1=1.0,
        bw=1.0,
        xshift=0.0,
         color="red",
        nbins=nothing,
        horizontal=false,
        normalize=false,
        fillalpha=.3,
         kwargs...
)
    kde = KDEUniv(ContinuousDim(), bw, X, MultiKDE.gaussian)

    nbins = isnothing(nbins) ? (Int64 ∘ round)(v1-v0) * 2 : 100
    x = Vector(LinRange(v0, v1, nbins))
    
    density = [MultiKDE.pdf(kde, _x, keep_all=false) for _x in x]
    density = normalize ? density ./ maximum(density) : density
    y = density .* scaling .+ shift
    
    if horizontal
        x, y = y, x
    end

    plot!(
        x .- xshift,
        y;  
        colorbar=false,
        color = color,
        lw=2,
        fillcolor=color, fillalpha=fillalpha, fillrange=zeros(length(density)) .+ shift .+  minimum(density),
        kwargs...
    )
end