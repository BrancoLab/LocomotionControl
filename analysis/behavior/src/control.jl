module control


using Ipopt
using InfiniteOpt

import Parameters: @with_kw
using IOCapture: IOCapture
import Term: Panel, RenderableText
import Term.style: apply_style
import MyterialColors: blue_light

import jcontrol: Track
import ..bicycle: Bicycle, State

export ModelVariables, ControlOptions, create_and_solve_control, Bounds, State
export DynamicsProblem, KinematicsProblem, realistict_control_options

abstract type MTMproblem end

⋅ = *

# ---------------------------------------------------------------------------- #
#                                    OPTIONS                                   #
# ---------------------------------------------------------------------------- #
struct Bounds{T<:Number}
    lower::T
    upper::T
end

"""
Constructor with an additional parametr to convert angles in degrees to radians
"""
Bounds(lower, upper, angle) = Bounds(deg2rad(lower), deg2rad(upper))

function Base.show(io::IO, b::Bounds)
    return print(
        io,
        apply_style(
            "[green]$(round(b.lower; digits=2))[/green] [red]≤[/red] var [red]≤[/red][green] $(round(b.upper; digits=2))[/green]",
        ),
    )
end

"""
Options for solving the control problem.
Include options for the optimizer itself as well 
other parameters such as bounds on allowed errors.
"""
@with_kw struct ControlOptions
    # errors bounds
    track_safety::Float64 = 1
    ψ_bounds::Bounds = Bounds(-30, 30, :angle)

    # control bounds
    u̇_bounds::Bounds = Bounds(-200, 200)
    δ̇_bounds::Bounds = Bounds(-50, 50, :angle)

    # varibles bounds
    u_bounds::Bounds = Bounds(5, 100)
    δ_bounds::Bounds = Bounds(-45, 45, :angle)
    ω_bounds::Bounds = Bounds(-500, 500, :angle)

    # dynamic system variables
    Fy_bounds::Bounds = Bounds(-100, 100)  # lateral forces
    Fu_bounds::Bounds = Bounds(-200, 200)  # driving force
    v_bounds::Bounds = Bounds(-125, 125)
end

realistict_control_options = ControlOptions(;
    u_bounds=Bounds(5, 80),
    u̇_bounds=Bounds(-180, 200),
    δ_bounds=Bounds(-50, 50, :angle),
    δ̇_bounds=Bounds(-4, 4),
    ω_bounds=Bounds(-400, 400, :angle),
    Fy_bounds=Bounds(-100, 100),
    v_bounds=Bounds(-125, 125),
)


"""
These are th best controls for the Dynamic model
as of 04/04/2022, they're also the very close
to the realistic values ranges.
"""
default_control_options = ControlOptions(;
u_bounds=Bounds(10, 90),
δ_bounds=Bounds(-80, 80, :angle),
δ̇_bounds=Bounds(-5, 5),
ω_bounds=Bounds(-600, 600, :angle),
v_bounds=Bounds(-15, 15),
Fu_bounds=Bounds(-3000, 4000),
)


# ---------------------------------------------------------------------------- #
#                                KINEMATIC MODEL                               #
# ---------------------------------------------------------------------------- #

"""
Model description.

Bicyle model with kinematics in the CoM reference frame.
    The CoM has position (x,y) and is at a distance l from
    the rear wheel, the total length is L. 
    The front wheel has a steering angle δ and there's a slip 
    angle β between the velocity vector at the CoM and the
    bike's orientation angle θ.

    Variables:
        L: total length
        l: rear wheel to CoM length

        x, y: position of CoM
        θ: orientation
        δ: steering angle
        β: slip angle of velocity vector
        u: velocity (speed)

    The model then has equations:
        ẋ = u * cos(θ + β)
        ẏ = u * sin(θ + β)
        θ̇ = ω = u * (tan(δ) * cos(β)) / L
    
    with:
        β = tan^{-1}(l * tan(δ)/L)

    There are two (bounded) controls:
        v̇
        δ̇

    The model is solved in the track's curvilinear coordinates system
    in which `n` represents the lateral distance from the center line 
    and `ψ` the angular error (θ - track angle). The kinamtics need
    to be expressed as a function of distance along the track `s` instead
    of time `t`, that's done with the scalig factor

        SF = (1 - n * κ(s))/(u * cos(ψ + β) + eps()) 

    where κ(s) represents the track's curvature. 

    Excluding x, y and θ's equation (because we are not doing things in the
    allocentric reference frame), we can write the model as:

        ∂(n, s) == SF * u * sin(ψ + β)
        ∂(ψ, s) == SF * ω - κ(s)
        
        ∂(u, s) == SF * v̇
        ∂(δ, s) == SF * δ̇
"""
struct KinematicsProblem <: MTMproblem end

"""
Create a `InfiniteOpt` model given a set of parameters and solve it to 
get the controls for the MTM problem.
"""
function create_and_solve_control(
    problem_type::KinematicsProblem,
    num_supports::Int,
    track::Track,
    bike::Bicycle,
    options::ControlOptions,
    initial_conditions::State,
    final_conditions::State;
    quiet::Bool=false,
    n_iter::Int=1000,
    tollerance::Float64=1e-10,
    verbose::Int=0,
)
    # initialize optimizer
    model = InfiniteModel(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", n_iter)
    set_optimizer_attribute(model, "acceptable_tol", tollerance)
    set_optimizer_attribute(model, "print_level", verbose)
    set_optimizer_attribute(model, "max_wall_time", 40.0)

    # register curvature function
    κ(s) = track.κ(s)
    @register(model, κ(s))

    # ----------------------------- define variables ----------------------------- #
    @infinite_parameter(model, s ∈ [0, track.S_f], num_supports = num_supports)

    @variables(
        model,
        begin
            # CONTROLS
            options.u̇_bounds.lower ≤ u̇ ≤ options.u̇_bounds.upper, Infinite(s)    # wheel acceleration
            options.δ̇_bounds.lower ≤ δ̇ ≤ options.δ̇_bounds.upper, Infinite(s)    # steering acceleration

            # track errors
            n, Infinite(s)  # constraints defined separtely
            options.ψ_bounds.lower ≤ ψ ≤ options.ψ_bounds.upper, Infinite(s)

            # other variables
            options.u_bounds.lower ≤ u ≤ options.u_bounds.upper, Infinite(s)
            options.δ_bounds.lower ≤ δ ≤ options.δ_bounds.upper, Infinite(s)

            β, Infinite(s)
            options.ω_bounds.lower ≤ ω ≤ options.ω_bounds.upper, Infinite(s)
            SF, Infinite(s)

            # time
            0 ≤ t ≤ 60, Infinite(s), (start = 10)
        end
    )

    # -------------------------- track width constraints ------------------------- #
    @parameter_function(model, allowed_track_width == track.width(s))
    @constraint(model, -allowed_track_width + bike.width ≤ n)
    @constraint(model, allowed_track_width - bike.width ≥ n)

    # ----------------------------- define kinematics ---------------------------- #        
    l = bike.l_r
    L = bike.L

    @constraints(
        model,
        begin
            β == atan(l * tan(δ) / L)
            ω == u * (tan(δ) * cos(β)) / L
            SF == (1 - n * κ(s)) / (u * cos(ψ + β) + eps())  # time -> space domain conversion factor

            ∂(n, s) == SF * u * sin(ψ + β)
            ∂(ψ, s) == SF * ω - κ(s)

            ∂(u, s) == SF * u̇
            ∂(δ, s) == SF * δ̇

            ∂(t, s) == SF
        end
    )

    # ----------------------- set initial/final conditions ----------------------- #
    @constraints(
        model,
        begin
            # initial conditions
            n(0) == initial_conditions.n
            ψ(0) == initial_conditions.ψ
            u(0) == initial_conditions.u
            δ(0) == initial_conditions.δ
            t(0) == 0
            β(0) == initial_conditions.β
            ω(0) == initial_conditions.ω

            # final conditions
            u(track.S_f) == final_conditions.u
            ω(track.S_f) == final_conditions.ω
        end
    )

    # --------------------------------- optimize --------------------------------- #
    # solve
    @objective(model, Min, ∫(SF, s))
    optimize!(model)

    # print info
    if !quiet
        c = IOCapture.capture() do
            println(solution_summary(optimizer_model(model)))
        end
        print(
            "\n" *
            Panel(
                RenderableText(c.output, "$blue_light italic");
                style="yellow1",
                title="IPoPT output",
                title_style="red bold",
                justify=:center,
            ) *
            "\n\n",
        )
    end

    return model
end

# ---------------------------------------------------------------------------- #
#                                DYNAMICS MODEL                                #
# ---------------------------------------------------------------------------- #
"""
The dynamic bicycle model is taken from:
    Dynamics and optimal control of road Vehicles (Massaro 2018)


The model has the following variables
    x, y:   position
    θ:      orientation
    δ:      steering angle
    u:      longitudinal velocity component
    v:      lateral velocity component
    ω:      angular velocity (around the CoM)
    β:      the slip angle of the velocity vector V
    
and these forces
    Fu:     logitudinal force applied to back tire
    Fr:     lateral force at the rear wheel
    Ff:     lateral force at the front wheel

The bike has the following constants
    l_f:     distance from CoM to front  wheel
    l_r:     distance from rear wheel to CoM
    m:       mass
    Iz:      angular momentum
    c:  tyre cornering stiffness

The two controls are 
    δ̇ and Fu


# --------------------------------- dynamics --------------------------------- #
The bike's dynamics are described by four equations of motion:
    δ̇    = δ̇   # the control
    mu̇   = mωv - Ff⋅sinδ + Fu
    mv̇   = -mωu + Ff⋅cosδ + Fr
    Iz⋅ω = l_f⋅Ff⋅cosδ - b⋅Fr


we can then compute the magnitude and slip angle of the velocity vector
    V̂ = √(u² + v²)
    β = arctan(v/u)

Finally the kinematics are given by:
    ẋ = V̂⋅cos(θ+β)
    ẏ = V̂⋅sin(θ+β)
    θ̇ = ω

# --------------------------------- solution --------------------------------- #
The strategy for the solution is the same as for the kinematics probelm.

Recast everything to the track's space using SF and using the track errors n and ψ,
like for the kinematics problem:
    ∂(n, s) == SF * u * sin(ψ + β)
    ∂(ψ, s) == SF * ω - κ(s)


With `SF` being the scaling factor:
    SF = (1 - n * κ(s))/(u * cos(ψ + β) + eps()) 
"""
struct DynamicsProblem <: MTMproblem end

"""
Create a `InfiniteOpt` model given a set of parameters and solve it to 
get the controls for the MTM problem.
"""
function create_and_solve_control(
    problem_type::DynamicsProblem,
    num_supports::Int,
    track::Track,
    bike::Bicycle,
    options::ControlOptions,
    initial_conditions::State,
    final_conditions::Union{Nothing, State, Symbol};
    quiet::Bool=false,
    n_iter::Int=1000,
    tollerance::Float64=1e-10,
    verbose::Int=0,
    α::Float64=1.0,
)
    
    # initialize optimizer
    model = InfiniteModel(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", n_iter)
    set_optimizer_attribute(model, "acceptable_tol", tollerance)
    set_optimizer_attribute(model, "print_level", verbose)
    set_optimizer_attribute(model, "max_wall_time", 60.0)

    # register curvature function
    κ(s) = track.κ(s)
    @register(model, κ(s))

    # ----------------------------- define variables ----------------------------- #
    @infinite_parameter(model, s ∈ [track.S[1], track.S[end]], num_supports = max(5, num_supports))

    @variables(
        model,
        begin
            # track errors
            n, Infinite(s)  # constraints defined separtely
            options.ψ_bounds.lower ≤ ψ ≤ options.ψ_bounds.upper, Infinite(s)

            # steering
            options.δ_bounds.lower ≤ δ ≤ options.δ_bounds.upper, Infinite(s)
            options.δ̇_bounds.lower ≤ δ̇ ≤ options.δ̇_bounds.upper, Infinite(s)    # control

            # long/lat/angular velocities
            options.u_bounds.lower ≤ u ≤ options.u_bounds.upper, Infinite(s)
            options.v_bounds.lower ≤ v ≤ options.v_bounds.upper, Infinite(s)
            options.ω_bounds.lower ≤ ω ≤ options.ω_bounds.upper, Infinite(s)

            # driving force
            options.Fu_bounds.lower ≤ Fu ≤ options.Fu_bounds.upper, Infinite(s)  # control 

            # time
            0 ≤ t ≤ 60, Infinite(s), (start = 10)   
       end
   )

    # -------------------------- track width constraints ------------------------- #
    @parameter_function(model, allowed_track_width == track.width(s))
    @constraint(model, -allowed_track_width + bike.width ≤ n)
    @constraint(model, allowed_track_width - bike.width ≥ n)

    # ----------------------------- define EOM       ---------------------------- #        
    l_r, l_f = bike.l_r, bike.l_f
    m, Iz, c = bike.m, bike.Iz, bike.c

    β = atan(v / (u))  # slip angle  
    V = √(u^2 + v^2)
    SF = (1 - n * κ(s)) / (V ⋅ cos(ψ + β))  # time -> space domain conversion factor

    Ff = c⋅(δ - (l_f⋅ω + v)/u)
    Fr = c⋅(l_r⋅ω - v)/u

    # get driving torque
    # τ = Fu / (V + eps())
    # τ = V > 50  ? Fu * .75 : Fu

    @constraints(
        model,
        begin
            # errors
            ∂(n, s) == SF * u ⋅ sin(ψ + β)
            ∂(ψ, s) == SF * ω - κ(s)

            # EOM
            ∂(δ, s) == SF * δ̇
            ∂(u, s) == SF / m * (m⋅ω⋅v - Ff⋅sin(δ) + Fu)
            ∂(v, s) == SF / m * (-m⋅ω⋅u + Ff⋅cos(δ) + Fr)
            ∂(ω, s) == SF / Iz * (l_f⋅Ff⋅cos(δ) - l_r⋅Fr)

            # time
            ∂(t, s) == SF
        end
    )

    # ----------------------- set initial/final conditions ----------------------- #
    # initial conditions
    @constraints(
        model,
        begin
            n(track.S[1]) == initial_conditions.n
            ψ(track.S[1]) == initial_conditions.ψ

            δ(track.S[1]) == initial_conditions.δ
            δ̇(track.S[1]) == initial_conditions.δ̇

            u(track.S[1]) == initial_conditions.u
            v(track.S[1]) == initial_conditions.v
            ω(track.S[1]) == initial_conditions.ω
            Fu(track.S[1]) == initial_conditions.Fu

            t(track.S[1]) == 0
        end
    )

    # final conditions
    if !isnothing(final_conditions)
        if final_conditions == :minimal
            @constraints(
                model, 
                begin
                n(track.S[end]) == 0
                ψ(track.S[end]) == 0

                end
            )
        else
            @constraints(
                model, 
                begin
                n(track.S[end]) == final_conditions.n
                ψ(track.S[end]) == final_conditions.ψ

                δ(track.S[end]) == final_conditions.δ
                δ̇(track.S[end]) == final_conditions.δ̇

                u(track.S[end]) == final_conditions.u
                v(track.S[end]) == final_conditions.v
                ω(track.S[end]) == final_conditions.ω
                Fu(track.S[end]) == final_conditions.Fu
                end
            )
            end
        end
    # --------------------------------- optimize --------------------------------- #
    # set_all_derivative_methods(model, FiniteDifference(Backward())) # less dependent on final conditions
    set_all_derivative_methods(model, OrthogonalCollocation(3))
    @objective(model, Min, ∫(α*SF + (1-α)*u, s))

    optimize!(model)

    # print info
    if !quiet
        c = IOCapture.capture() do
            println(solution_summary(optimizer_model(model)))
        end
        print(
            "\n" *
            Panel(
                c.output;
                style="yellow1",
                title="IPoPT output",
                title_style="red bold",
                justify=:center,
            ) *
            "\n\n",
        )
    end
    return model
end

end
