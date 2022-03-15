module forwardmodel
import InfiniteOpt: value
using DifferentialEquations
import DifferentialEquations: solve as desolve
using Interpolations: Interpolations
import Parameters: @with_kw

import jcontrol: Track, ξ
import ..control: KinematicsProblem, DynamicsProblem
import ..bicycle: State, Bicycle

export Solution, run_forward_model

@with_kw struct Solution
    δt::Float64
    t::Vector{Float64}    # time
    x::Vector{Float64}    # position
    y::Vector{Float64}
    θ::Vector{Float64}    # orientation
    δ::Vector{Float64}    # steering angle
    u::Vector{Float64}    # velocity  | fpr DynamicsProblem its the longitudinal velocity component
    
    # Dynamics problem only
    v::Vector{Float64} = Vector{Float64}[]  # lateral velocity
    ω::Vector{Float64} = Vector{Float64}[]  # angular velocity

    # controls
    u̇::Vector{Float64}
    δ̇::Vector{Float64}
end

"""
The solution to the control problem gives the history of controls u̇ and δ̇ 
as a function of `s`. `run_forward_model` takes these and integrates the model's ODEs to 
obtain the trajectory as a function of time `t`.
"""
function run_forward_model end

"""
Compute solution to forward problem
"""
function run_forward_model(mtm_solution, kinematics!, u0, δt)
    @info "Starting forward integration"
    """
        Get the control values at a moment in time.

        This is done by looking at the values of TIME in the MTM
        solution and finding the index of the time step closest to the current time.
    """
    function controls(t)
        if t == t_start
            return u̇[1], δ̇[1]
        elseif t == t_end
            return u̇[end], δ̇[end]
        else
            idx = findfirst(TIME .>= t)    
            return u̇[idx], δ̇[idx]
        end
    end


    # get variables
    TIME = value(mtm_solution[:t]) # time in the MTM model's solution
    t_start, t_end = TIME[1], TIME[end]
    # time = collect(0:δt:(TIME[end] + δt))  # time in ODEs solution

    # controls from MTM solution
    u̇ = value(mtm_solution[:u̇])
    δ̇ = value(mtm_solution[:δ̇])

    # fix controls (finite problem approx introduces boundary errors)
    u̇[1], u̇[end] = u̇[2], u̇[end-1]
    δ̇[1], δ̇[end] = δ̇[2], δ̇[end-1]

    # solve problem
    tspan = (0.0, TIME[end])
    prob = ODEProblem(kinematics!, u0, tspan, t -> controls(t))
    sol = desolve(prob, Tsit5(); reltol=1e-10, abstol=1e-10, saveat=δt)
    @info "ODEs solution found" sol.t[end]
    return sol, controls
end


"""
Forward integration for KinematicsProblem
"""
function run_forward_model(problemtype::KinematicsProblem,  mtm_solution, x_0::State, bike::Bicycle; δt=0.1)
    """
    Bicicle model kinematics equatiosn
    """
    l = bike.l_r
    L = bike.l_f + bike.l_r
    function kinematics!(du, x, p, t)
        u̇t, δ̇t = p(t)     # controls
        _, _, θ, δ, u = x   # state

        β = atan(l * tan(δ) / L)

        du[1] = u * cos(θ + β)              # ẋ = u cos(θ + β)
        du[2] = u * sin(θ + β)              # ẏ = u sin(θ + β)
        du[3] = u * (tan(δ) * cos(β)) / L   # θ̇ = u * (tan(δ) * cos(β)) / L
        du[4] = δ̇t                          # δ̇
        du[5] = u̇t                          # u̇
        return du
    end

    # initial condition
    u0 = [
        x_0.x,  # x
        x_0.y,  # y
        x_0.θ,  # θ
        x_0.δ,  # δ
        x_0.u   # u
    ]

    # solve
    sol, controls = run_forward_model(mtm_solution, kinematics!, u0, δt)

    # return solution
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



"""
Forward integration for DynamicsProblem
"""
function run_forward_model(problemtype::DynamicsProblem,  mtm_solution, x_0::State, bike::Bicycle; δt=0.1)
    """
    Bicicle model kinematics equatiosn
    """
    l_r, l_f = bike.l_r, bike.l_f
    cf, cr = bike.cf, bike.cr
    m, Iz = bike.m, bike.Iz

    function kinematics!(du, x, p, t)
        u̇t, δ̇t = p(t)     # controls
        _, _, θ, δ, u, ω, v = x   # state

        # compute slip angles
        αf = δ - atan((v + ω * l_f)/u)
        αr = δ - atan((v - ω * l_r)/u)

        # compute lateral forces
        Ff = cf * αf
        Fr = cr * αr

        # compute state variabels
        du[1] = u * cos(θ) + v * sin(θ)       # ẋ
        du[2] = u * sin(θ) + v * cos(θ)       # ẏ
        du[3] = ω                             # θ̇
        du[4] = δ̇t                            # δ̇
        du[5] = u̇t                            # u̇
        du[6] = (Ff * l_f - Fr * l_r) / Iz    # ω̇
        du[7] = (Ff + Fr - u * ω) / m         # v̇
        return du
    end

    # initial condition
    u0 = [
        x_0.x,  # x
        x_0.y,  # y
        x_0.θ,  # θ
        x_0.δ,  # δ
        x_0.u,  # u
        x_0.ω,  # ω
        x_0.v,  # v
    ]

    # solve
    sol, controls = run_forward_model(mtm_solution, kinematics!, u0, δt)

    # return solution
    return Solution(;
        δt=δt,
        t=sol.t,
        x=[u[1] for u in sol.u],
        y=[u[2] for u in sol.u],
        θ=[u[3] for u in sol.u],
        δ=[u[4] for u in sol.u],
        u=[u[5] for u in sol.u],
        ω=[u[6] for u in sol.u],        
        v=[u[7] for u in sol.u],
        u̇=vcat([c[1] for c in controls.(sol.t)]),
        δ̇=vcat([c[2] for c in controls.(sol.t)]),
    )
end

end
