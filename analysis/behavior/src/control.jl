module control

using InfiniteOpt, Ipopt
import InfiniteOpt: set_optimizer_attribute
import Parameters: @with_kw

import jcontrol: Track
import ..bicycle: Bicycle, State

export ModelVariables, ControlOptions, create_and_solve_control, Bounds, State

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
        v: velocity (speed)

    The model then has equations:
        ẋ = v * cos(θ + β)
        ẏ = v * sin(θ + β)
        θ̇ = ω = v * (tan(δ) * cos(β)) / L
    
    with:
        β = tan^{-1}(l * tan(δ)/L)

    There are two (bounded) controls:
        uv = v̇
        uδ = δ̇



    The model is solved in the track's curvilinear coordinates system
    in which `n` represents the lateral distance from the center line 
    and `ψ` the angular error (θ - track angle). The kinamtics need
    to be expressed as a function of distance along the track `s` instead
    of time `t`, that's done with the scalig factor

        SF = (1 - n * κ(s))/(v * cos(ψ + β) + eps()) 

    where κ(s) represents the track's curvature. 

    Excluding x, y and θ's equation (because we are not doing things in the
    allocentric reference frame), we can write the model as:

        ∂(n, s) == SF * v * sin(ψ + β)
        ∂(ψ, s) == SF * ω - κ(s)
        
        ∂(v, s) == SF * uv
        ∂(δ, s) == SF * uδ
"""

# ---------------------------------------------------------------------------- #
#                                    OPTIONS                                   #
# ---------------------------------------------------------------------------- #
@with_kw struct Bounds
    lower::Number
    upper::Number
end

"""
Constructor with an additional parametr to convert angles in degrees to radians
"""
Bounds(lower, upper, angle) = Bounds(deg2rad(lower), deg2rad(upper))

"""
Options for solving the control problem.
Include options for the optimizer itself as well 
other parameters such as bounds on allowed errors.
"""
@with_kw struct ControlOptions
    n_iter::Int = 1000
    num_supports::Int = 100
    tollerance::Number = 1e-8
    verbose::Int = 0

    # errors bounds
    track_safety::Float64 = 0.9
    ψ_bounds::Bounds = Bounds(-π / 2, π / 2)

    # control bounds
    uv_bounds::Bounds = Bounds(-200, 200)
    uδ_bounds::Bounds = Bounds(-50, 50)

    # varibles bounds
    v_bounds::Bounds = Bounds(0, 100)
    δ_bounds::Bounds = Bounds(deg2rad(-45), deg2rad(45))
end

# ---------------------------------------------------------------------------- #
#                                 CONTROL MODEL                                #
# ---------------------------------------------------------------------------- #
"""
Create a `InfiniteOpt` model given a set of parameters and solve it to 
get the controls for the MTM problem.
"""
function create_and_solve_control(
    track::Track,
    bike::Bicycle,
    options::ControlOptions,
    initial_conditions::State,
    final_conditions::State,
)
    # initialize optimizer
    model = InfiniteModel(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.n_iter)
    set_optimizer_attribute(model, "acceptable_tol", options.tollerance)
    set_optimizer_attribute(model, "print_level", options.verbose)

    # register curvature function
    κ(s) = track.κ(s)
    @register(model, κ(s))

    # ----------------------------- define variables ----------------------------- #
    @infinite_parameter(model, s ∈ [0, track.S_f], num_supports = options.num_supports)

    allowed_track_width = (track.width - bike.width/2) * options.track_safety
    @variables(
        model,
        begin
            # CONTROLS
            options.uv_bounds.lower ≤ uv ≤ options.uv_bounds.upper, Infinite(s)    # wheel acceleration
            options.uδ_bounds.lower ≤ uδ ≤ options.uδ_bounds.upper, Infinite(s)    # steering acceleration

            # track errors
            -allowed_track_width ≤ n ≤ allowed_track_width, Infinite(s)
            options.ψ_bounds.lower ≤ ψ ≤ options.ψ_bounds.upper, Infinite(s)

            # other variables
            options.v_bounds.lower ≤ v ≤ options.v_bounds.upper, Infinite(s)
            options.δ_bounds.lower ≤ δ ≤ options.δ_bounds.upper, Infinite(s)

            β, Infinite(s)
            ω, Infinite(s)
            SF, Infinite(s)

            # time
            0 ≤ t ≤ 60, Infinite(s), (start = 10)
        end
    )

    # ----------------------------- define kinematics ---------------------------- #        
    l, L = bike.l, bike.L

    @constraints(
        model,
        begin
            β == atan(l * tan(δ) / L)
            ω == v * (tan(δ) * cos(β)) / L
            SF == (1 - n * κ(s)) / (v * cos(ψ + β) + eps())  # time -> space domain conversion factor

            ∂(n, s) == SF * v * sin(ψ + β)
            ∂(ψ, s) == SF * ω - κ(s)

            ∂(v, s) == SF * uv
            ∂(δ, s) == SF * uδ

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
            v(0) == initial_conditions.v
            δ(0) == initial_conditions.δ
            t(0) == 0
            β(0) == initial_conditions.β
            ω(0) == initial_conditions.ω

            # final conditions
            v(track.S_f) == final_conditions.v
            ω(track.S_f) == final_conditions.ω
            n(track.S_f) == 0
            ψ(track.S_f) == 0
        end
    )

    # --------------------------------- optimize --------------------------------- #
    @info "control model ready, solving with IPOPT" options.num_supports options.n_iter
    @objective(model, Min, ∫(SF, s))
    optimize!(model)

    @info "Model optimization complete" termination_status(model) objective_value(model) value(
        model[:t]
    )[end]

    # done
    return model
end

end
