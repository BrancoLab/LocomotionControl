# ---------------------------------------------------------------------------- #
#                                 FETCHING DATA                                #
# ---------------------------------------------------------------------------- #

py"""

def get_bouts(direction, baseline = True):
    bouts =  pd.DataFrame(
        SessionCondition * LocomotionBouts * ProcessedLocomotionBouts & f"direction='{direction}'" & "corrected_start_frame > 0"
    )
    if baseline:
        bouts = bouts.loc[bouts.condition == "naive"]
    else:
        bouts = bouts.loc[bouts.condition != "naive"]

    return bouts

"""

"""
Convert PyObjects to corresponding Julia types.
"""
function py2jl end

py2jl(x) = try
    [py2jl(y) for y in x] |> collect
catch
    try
        convert(Float64, x)
    catch
        x
    end
end
py2jl(x::Union{Number, AbstractString}) = x


"""
Convert a PyObject of a pd.DataFrame to a julia DataFrame
"""
function Base.convert(::Type{DataFrame}, obj::PyCall.PyObject)
    out = Dict()
    for col in obj["columns"]
        v = py2jl(obj[col]["values"])
        out[col] = v
    end
    return out |> DataFrame
end



# ---------------------------------------------------------------------------- #
#                               DATA MANIPULATION                              #
# ---------------------------------------------------------------------------- #

"""
    stack_kinematic_variables(bouts)
"""
function stack_kinematic_variables(bouts)
    X = vcat(
        bouts[:, "x"]...
    )
    X[isnan.(X)] .= 0

    Y = vcat(
        bouts[:, "y"]...
    )
    Y[isnan.(Y)] .= 0

    S = vcat(
        bouts[:, "speed"]...
    )
    S[isnan.(S)] .= 0

    A = vcat(
        bouts[:, "acceleration"]...
    )
    A[isnan.(A)] .= 0

    T = vcat(
        bouts[:, "angvel"]...
    )
    T[isnan.(T)] .= 0

    D = vcat(
        bouts[:, "angaccel"]...
    )
    D[isnan.(D)] .= 0

    return X, Y, S, A, T, D
end



"""
    get_roi_bouts(roi::Int, bouts)

Trim the bouts to the region of interest.
"""
function get_roi_bouts(roi::Int, bouts)
    roi_bouts = copy(bouts)
    for (i, bout) in enumerate(eachrow(roi_bouts))
        start = findfirst(bout.s .> roi_limits[roi][1])
        stop = findfirst(bout.s .> roi_limits[roi][2])

        roi_bouts[i, :x] = bout.x[start:stop]
        roi_bouts[i, :y] = bout.y[start:stop]
        roi_bouts[i, :s] = bout.s[start:stop]
        roi_bouts[i, :speed] = bout.a[start:stop]
        roi_bouts[i, :acceleration] = bout.a[start:stop]
        roi_bouts[i, :angvel] = bout.t[start:stop]
        roi_bouts[i, :angaccel] = bout.d[start:stop]
    end

    return roi_bouts
end

"""
Given a set of coordiantes, bin them in 2D
and then get the average of the Z values in each bin's time points.
"""
function bin_z_by_xy(
        X::Vector, Y::Vector, Z::Vector; 
        stepsize::Number=1,
        x₀::Number=0, y₀::Number=0,
        x₁::Number=40, y₁::Number=60,
    )::Tuple{Vector, Vector, Matrix}

    x_edges = range(x₀, x₁, step=stepsize) |> collect
    y_edges = range(y₀, y₁, step=stepsize) |> collect
    z = zeros(length(x_edges)-1, length(y_edges)-1)
    
    for i in 2:length(x_edges), j in 2:length(y_edges)
        in_bin = (X .> x_edges[i-1]) .& (X .< x_edges[i]) .& (Y .> y_edges[j-1]) .& (Y .< y_edges[j])
        z[i-1, j-1] = mean(Z[in_bin])
    end
    
    x = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2
    y = (y_edges[1:end-1] .+ y_edges[2:end]) ./ 2
    z = z' |> Matrix;
    return x, y, z
end



# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #

"""
    plot_bouts_trace

Plot the XY coordinates of each bout as a trace.
"""
function plot_bouts_trace(bouts; kwargs...)
    plt = plot(
        ; kwargs..., arena_ax_kwargs...
    )
    plot_bouts_trace!(plt, bouts)
end

function plot_bouts_trace!(plt, bouts)
     for bout in eachrow(bouts)
         plot!(plt, bout.x, bout.y, label=nothing, lw=.5, color=:grey)
     end

     return plt
end


"""
Plot 2D heatmaps of kinematic variable values over arena locations
"""
function kinematics_heatmaps(X, Y, S, A, T, D; Δ=10, stepsize=2.5, kwargs...)
    x, y, s = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], S[1:Δ:end]; stepsize=stepsize)
    p1 = heatmap(x, y, s; clims=(0, 80), colorbar_title="cm/s", title="speed", arena_ax_kwargs...)

    x, y, a = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], A[1:Δ:end]; stepsize=stepsize)
    p2 = heatmap(x, y, a; color=:bwr, clims=(-3, 3), colorbar_title="cm/s²",  title="acceleration", arena_ax_kwargs...)

    x, y, t = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], T[1:Δ:end]; stepsize=stepsize)
    p3 = heatmap(x, y, t;color=:PRGn, clims=(-500, 500), colorbar_title = "°/s",  title="angvel", arena_ax_kwargs...)

    x, y, d = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], D[1:Δ:end]; stepsize=stepsize)
    p4 = heatmap(x, y, d; color=:PRGn, clims=(-75, 75), colorbar_title="°/s²", title="angaccel", arena_ax_kwargs...)


    return plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800); kwargs...)
end




plot_density(x, args...; kwargs...) = plot_density!(plot(grid=false,), x, args...; kwargs...)

"""
A KDE density plot with additional customization and quantile lines.
"""
function plot_density!(p, x, args...; y=0, npoints=2500, boundary=nothing, bandwidth=nothing, color=:black, kwargs...)
    boundary = isnothing(boundary) ? quantile(x, (0.01, 0.99)) : boundary
    bandwidth = isnothing(bandwidth) ? (boundary[1]-boundary[2])/100 : bandwidth
    k = kde(x; npoints=npoints, boundary=boundary, bandwidth=bandwidth)

    q1, q25, q75, q99 = quantile(x, (0.01, 0.25, 0.75, 0.99))
    μ = mean(x)


    plot!(p,
        [q1, q99], [y, y], lw=3, color=color, label=nothing, alpha=.75
    )
    plot!(p,
        [q25, q75], [y, y], lw=4, color=color, label=nothing,
    )
    scatter!(p,
        [μ], [y], ms=8, color=:white, label=nothing, msw=2, msc=:black
    )

    plot!(p,
        k.x, k.density,
        color=color, fill=(0, 0.2, color),
        args...;
        kwargs...
    )
end