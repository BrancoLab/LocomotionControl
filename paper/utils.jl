# ---------------------------------------------------------------------------- #
#                                 FETCHING DATA                                #
# ---------------------------------------------------------------------------- #

py"""

def get_bouts(direction, baseline = True):
    bouts =  pd.DataFrame(
        SessionCondition * LocomotionBouts * ProcessedLocomotionBouts & f"direction='{direction}'" & "duration<12"
    )
    if baseline:
        bouts = bouts.loc[bouts.condition == "naive"]
    else:
        bouts = bouts.loc[bouts.condition != "naive"]

    return bouts

"""

# ---------------------------------------------------------------------------- #
#                               DATA MANIPULATION                              #
# ---------------------------------------------------------------------------- #

"""
    stack_kinematic_variables(bouts)
"""
function stack_kinematic_variables(bouts)
    X = vcat(
        getfield.(bouts, :x)...
    )
    Y = vcat(
        getfield.(bouts, :y)...
    )
    S = vcat(
        getfield.(bouts, :speed)...
    )
    A = vcat(
        getfield.(bouts, :acceleration)...
    )
    T = vcat(
        getfield.(bouts, :angvel)...
    )
    D = vcat(
        getfield.(bouts, :angaccel)...
    )

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


# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #

"""
    plot_bouts_trace

Plot the XY coordinates of each bout as a trace.
"""
function plot_bouts_trace(bouts; kwargs...)
    plt = plot(
        ;kwargs..., arena_ax_kwargs...
    )
    plot_bouts_trace!(plt, bouts)
end

function plot_bouts_trace!(plt, bouts)
     for bout in eachrow(bouts)
         plot!(plt, bout[:x], bout[:y], label=nothing, lw=.5, color=:grey)
     end
end


function kinematics_heatmaps(X, Y, S, A, T, D)
    p1 = contourf(
        X, Y, S, levels=range(0, 100, length=10), title="speed"; arena_ax_kwargs...
    )
    p2 = contourf(
        X, Y, A, levels=range(-100, 100, length=10), title="acceleration"; arena_ax_kwargs...
    )
    p3 = contourf(
        X, Y, T, levels=range(-100, 100, length=10), title="angvel"; arena_ax_kwargs...
    )
    p4 = contourf(
        X, Y, D, levels=range(-100, 100, length=10), title="angaccel"; arena_ax_kwargs...
    )

    return plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800))
end