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
    x::Vector    # position
    y::Vector
    θ::Vector    # orientation
    δ::Vector    # steering angle
    u::Vector    # velocity  | fpr DynamicsProblem its the longitudinal velocity component
    
    # Dynamics problem only
    v::Vector = []  # lateral velocity

    # controls
    u̇::Vector
    δ̇::Vector
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
    l = bike.l_r
    L = bike.l_r + bike.l_f

    """
        Get the control values at a moment in time.

        This is done by looking at the values of TIME in the MTM
        solution and finding the index of the time step closest to the current time.
    """
    function controls(t)
        if t == 0
            return u̇[1], δ̇[1]
        end

        # get support indes
        t_large = findfirst(TIME .>= t)

        if TIME[t_large] - t == 0
            return u̇[t_large], δ̇[t_large]
        end

        # get controls
        t_idx_post = isnothing(t_large) ? length(time) : t_large
        return u̇[t_idx_post], δ̇[t_idx_post]
        idx = n(t)
        return u̇(idx), δ̇(idx)
    end

    """
    Bicicle model kinematics equatiosn
    """
    function kinematics!(du, x, p, t)
        u̇t, δ̇t = p(t)     # controls
        _, _, θ, δ, u = x   # state

        β = atan(l * tan(δ) / L)

        du[1] = u * cos(θ + β)              # ẋ = u cos(θ + β)
        du[2] = u * sin(θ + β)              # ẏ = u sin(θ + β)
        du[3] = u * (tan(δ) * cos(β)) / L   # θ̇ = u * (tan(δ) * cos(β)) / L
        du[4] = δ̇t                          # δ̇
        du[5] = u̇t                          # vu̇
        return du
    end

    # get variables
    TIME = value(mtm_solution[:t]) # time in the MTM model's solution
    time = collect(0:δt:(TIME[end] + δt))  # time in ODEs solution

    # controls from MTM solution
    u̇ = value(mtm_solution[:u̇])
    δ̇ = value(mtm_solution[:δ̇])

    # initial state
    u0 = [x_0.x, x_0.y, x_0.θ, x_0.δ, x_0.u]

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
        u=[u[5] for u in sol.u],
        u̇=vcat([c[1] for c in controls.(sol.t)]),
        δ̇=vcat([c[2] for c in controls.(sol.t)]),
    )
end

end
