module forwardmodel
import InfiniteOpt: value
using DifferentialEquations
import DifferentialEquations: solve as desolve
using Interpolations: Interpolations
import Parameters: @with_kw

import jcontrol: Track, ξ
import ..bicycle: State, Bicycle

export Solution, run_forward_model

@with_kw struct Solution
    δt::Float64
    t::Vector
    x::Vector
    y::Vector
    θ::Vector
    δ::Vector
    v::Vector
    uv::Vector
    uδ::Vector
end

"""
Given a solution to the MTM control problem in the track's 
coordinates system (`model`), soves the bicycle model ODEs
given the values of the controls from the MTM solution to
get the solution in time and in the allocentric reference frame,

# Arguments
- x_0: initial conditions
- δt: time interval at which the ODEs solution are saved
"""
function run_forward_model(mtm_solution, x_0::State, bike::Bicycle; δt=0.1)
    @info "Starting forward integration"
    L = bike.L
    l = bike.l

    """
        Get the control values at a moment in time.

        This is done by looking at the values of TIME in the MTM
        solution and finding the index of the time step closest to the current time.
    """
    function controls(t)
        # return uv(t), uδ(t)
        if t == 0
            return uv[1], uδ[1]
        end

        # get support indes
        t_large = findfirst(TIME .>= t)

        if TIME[t_large] - t == 0
            return uv[t_large], uδ[t_large]
        end

        # get controls
        t_idx_post = isnothing(t_large) ? length(time) : t_large
        return uv[t_idx_post], uδ[t_idx_post]
        idx = n(t)
        return uv(idx), uδ(idx)
    end

    """
    Bicicle model kinematics equatiosn
    """
    function kinematics!(du, u, p, t)
        uvt, uδt = p(t)     # controls
        _, _, θ, δ, v = u   # state

        β = atan(l * tan(δ) / L)

        du[1] = v * cos(θ + β)              # ẋ = v cos(θ + β)
        du[2] = v * sin(θ + β)              # ẏ = v sin(θ + β)
        du[3] = v * (tan(δ) * cos(β)) / L     # θ̇ = v * (tan(δ) * cos(β)) / L
        du[4] = uδt                         # δ̇ = uδ
        du[5] = uvt                         # v̇ = uv
        return du
    end

    # get variables
    TIME = value(mtm_solution[:t]) # time in the MTM model's solution
    T = TIME[end]
    N = length(TIME)
    time = collect(0:δt:(TIME[end] + δt))  # time in ODEs solution

    # controls from MTM solution
    """
        These functions take a value t ∈ [0 T], 
        and use it to get the corresponding value out of the 
        interpolated vector ξ(x) by getting the index
        value n ∈ [0 N] ⊂ R corresponding to t.
    """
    # time = range(0, T, length=length(TIME))
    # uv =  Interpolations.scale(ξ(value(mtm_solution[:uv])), time)
    # uδ =  Interpolations.scale(ξ(value(mtm_solution[:uδ])), time)

    uv = value(mtm_solution[:uv])
    uδ = value(mtm_solution[:uδ])

    # initial state
    u0 = [x_0.x, x_0.y, x_0.θ, x_0.δ, x_0.v]

    # solve problem
    tspan = (0.0, TIME[end])
    prob = ODEProblem(kinematics!, u0, tspan, t -> controls(t))
    sol = desolve(prob, Tsit5(); reltol=1e-12, abstol=1e-12, saveat=δt)
    @info "ODEs solution found" sol.t[end]

    return Solution(;
        δt=δt,
        t=sol.t,
        x=[u[1] for u in sol.u],
        y=[u[2] for u in sol.u],
        θ=[u[3] for u in sol.u],
        δ=[u[4] for u in sol.u],
        v=[u[5] for u in sol.u],
        uv=vcat([c[1] for c in controls.(sol.t)]),
        uδ=vcat([c[2] for c in controls.(sol.t)]),
    )
end

end
