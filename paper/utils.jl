# ---------------------------------------------------------------------------- #
#                                 FETCHING DATA                                #
# ---------------------------------------------------------------------------- #

py"""
def get_bouts(direction, baseline = True, target="MOs"):
    if baseline:
        bouts =  pd.DataFrame(
            SessionCondition * LocomotionBouts * ProcessedLocomotionBouts & f"direction='{direction}'" & "corrected_start_frame > 0"
        )
        bouts = bouts.loc[bouts.condition == "naive"]
    else:
        bouts =  pd.DataFrame(
            SessionCondition * LocomotionBouts * Surgery *  ProcessedLocomotionBouts  & 
                    f"direction='{direction}'" & "corrected_start_frame > 0" & f"target = '{target}'"
        )
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
    moving_average(A::AbstractArray, m::Int)

Moving average smoothing.
"""
function moving_average(A::AbstractArray, m::Int)
    out = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = m ÷ 2 * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J = max(Ifirst, I - I1):min(Ilast, I + I1)
            s += A[J]
            n += 1
        end
        out[I] = s / n
    end
    return out
end



"""
    stack_kinematic_variables(bouts)
"""
function stack_kinematic_variables(bouts)
    s = vcat(
        bouts[:, "s"]...
    )
    s[isnan.(s)] .= 0

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

    return s, X, Y, S, A, T, D
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
        
        #  && continue
        z[i-1, j-1] = sum(in_bin) < 5 ? NaN :  mean(Z[in_bin])
    end
    
    x = (x_edges[1:end-1] .+ x_edges[2:end]) ./ 2
    y = (y_edges[1:end-1] .+ y_edges[2:end]) ./ 2
    z = z' |> Matrix;
    return x, y, z
end

"""
Given two vectors, bin the first and get all time points in each bin, return
the mean and std of the second vector in each bin.
"""
function bin_x_by_y(x::Vector, y::Vector; x₀=minimum(x), x₁=maximum(x), stepsize=1)
    bins = x₀:stepsize:x₁ |> collect
    bins_centers = bins[1:end-1] .+ stepsize/2
    y_binned = zeros(length(bins)-1)
    y_binned_std = zeros(length(bins)-1)

    for (i, bin) in enumerate(bins[1:end-1])
        y_binned[i] = mean(y[(x .> bin) .& (x .<= bins[i+1])])
        y_binned_std[i] = std(y[(x .> bin) .& (x .<= bins[i+1])])
    end
    return bins_centers, y_binned, y_binned_std
end


"""
    get_roi_bouts(roi::Int, bouts)

Trim the bouts to the region of interest.
"""
function get_roi_bouts(roi::Int, bouts)
    roi_bouts = Dict(
        "x" => [],
        "y" => [],
        "s" => [],
        "speed" => [],
        "acceleration" => [],
        "angvel" => [],
        "angaccel" => [],
    )

    for (i, bout) in enumerate(eachrow(bouts))
        start = findfirst(bout.s .>= roi_limits[roi][1])
        stop = findlast(bout.s .<= roi_limits[roi][2])
        
        push!(roi_bouts["x"], bout.x[start:stop])
        push!(roi_bouts["y"], bout.y[start:stop])
        push!(roi_bouts["s"], bout.s[start:stop])
        push!(roi_bouts["speed"], bout.speed[start:stop])
        push!(roi_bouts["acceleration"], bout.acceleration[start:stop])
        push!(roi_bouts["angvel"], bout.angvel[start:stop])
        push!(roi_bouts["angaccel"], bout.angaccel[start:stop])
    end
    return DataFrame(roi_bouts)
end



# ---------------------------------------------------------------------------- #
#                                  KINEMATICS                                  #
# ---------------------------------------------------------------------------- #

"""
Get the time points in which the mouse starts slowing down 
and turning in a given bout (assumed to be trimmed around an ROI).
"""
function get_bout_slow_turn_onsets(bout)
    speed = moving_average(bout.speed, 5)
    max_speed = maximum(speed)
    atmax = min(argmax(speed), length(speed)-1)
    after_peak = speed[atmax:end]

    slow = nothing
    th = 0.90
    while isnothing(slow) && th > 0.5
        onset = findfirst(after_peak .<= th*max_speed)
        th -= 0.1
    end
    slow = something(slow, 1) + atmax
    
    

    ω = moving_average(abs.(bout.angvel), 5)
    ω = ω[slow:end]

    turn, th = nothing, 0.25
    while isnothing(turn) && th > 0
        turn = findfirst(ω .> th*maximum(ω))
        th -= 0.05
    end

    return slow, turn+slow
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
        ; arena_ax_kwargs...
    )
    plot_bouts_trace!(plt, bouts; kwargs...,)
end

function plot_bouts_trace!(plt, bouts; color=:grey, label=nothing, kwargs...)
     for (i, bout) in enumerate(eachrow(bouts))
         plot!(plt, bout.x, bout.y,  
            lw=.5, color=color,
            label= i == 1 ? label : nothing; kwargs...)
     end

     return plt
end


"""
Plot 2D heatmaps of kinematic variable values over arena locations
"""
function kinematics_heatmaps(X, Y, S, A, T, D; Δ=1, stepsize=1.5, kwargs...)
    x, y, s = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], S[1:Δ:end]; stepsize=stepsize)
    p1 = heatmap(x, y, s; clims=(20, 80), colorbar_title="cm/s", title="speed", arena_ax_kwargs...)

    x, y, a = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], A[1:Δ:end]; stepsize=stepsize)
    p2 = heatmap(x, y, a; color=:bwr, clims=(-3, 3), colorbar_title="cm/s²",  title="acceleration", arena_ax_kwargs...)

    x, y, t = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], T[1:Δ:end]; stepsize=stepsize)
    p3 = heatmap(x, y, t;color=:PRGn, clims=(-500, 500), colorbar_title = "°/s",  title="angvel", arena_ax_kwargs...)

    x, y, d = bin_z_by_xy(X[1:Δ:end], Y[1:Δ:end], D[1:Δ:end]; stepsize=stepsize)
    p4 = heatmap(x, y, d; color=:PRGn, clims=(-50, 50), colorbar_title="°/s²", title="angaccel", arena_ax_kwargs...)


    return plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800); kwargs...)
end




plot_density(x, args...; kwargs...) = plot_density!(plot(grid=false,), x, args...; kwargs...)

"""
A KDE density plot with additional customization and quantile lines.
"""
function plot_density!(p, x, args...; y=0, npoints=2500, boundary=nothing, bandwidth=nothing, color=:black, kwargs...)
    boundary = isnothing(boundary) ? quantile(x, (0.01, 0.99)) : boundary
    bandwidth = isnothing(bandwidth) ? (boundary[2]-boundary[1])/50 : bandwidth
    k = kde(x; npoints=npoints, boundary=boundary, bandwidth=bandwidth)

    q1, q25, q75, q99 = quantile(x, (0.01, 0.25, 0.75, 0.99))
    μ = mean(x)


    plot!(p,
        [q1, q99], [y, y], lw=3, color=color, label=nothing, alpha=.75
    )
    plot!(p,
        [q25, q75], [y, y], lw=6, color=color, label=nothing,
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