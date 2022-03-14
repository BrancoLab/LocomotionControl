module control

using InfiniteOpt, Ipopt
import InfiniteOpt: set_optimizer_attribute
import Parameters: @with_kw

import jcontrol: Track
import ..bicycle: Bicycle, State

export ModelVariables, ControlOptions, create_and_solve_control, Bounds, State
export DynamicsProblem, KinematicsProblem, realistict_control_options


abstract type MTMproblem end

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
    u̇_bounds::Bounds = Bounds(-200, 200)
    δ̇_bounds::Bounds = Bounds(-50, 50)

    # varibles bounds
    u_bounds::Bounds = Bounds(0, 100)
    δ_bounds::Bounds = Bounds(deg2rad(-45), deg2rad(45))
end


realistict_control_options = Dict(
    "u" => Bounds(5, 80),
    "u̇" => Bounds(-150, 200),
    "δ" => Bounds(-20, 20, :angle),
    "δ̇" => Bounds(-80, 80, :angle),
    # "ω" => Bounds(-450, 450, :angle)
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
            ω, Infinite(s)
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
    L = bike.l_r + bike.l_f

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




# ---------------------------------------------------------------------------- #
#                                DYNAMICS MODEL                                #
# ---------------------------------------------------------------------------- #
"""
The dynamic bicycle model is taken from:
    https://thef1clan.com/2020/12/23/vehicle-dynamics-the-dynamic-bicycle-model/


The model has the following variables
    x, y:   position
    θ:      orientation
    δ:      steering angle
    u:      longitudinal velocity component
    v:      lateral velocity component
    ω:      angualr velocity (around the CoM)

The bike has the following constants
    l_f:     distance from CoM to front  wheel
    l_r:     distance from rearh wheel to CoM
    m:       mass
    Iz:      angular momentum
    cf, cr:  tyre cornering stiffness

The two controls are 
    u̇, δ̇


# --------------------------------- dynamics --------------------------------- #
To desribe the bike's movement we need to defien v̇ and ω̇ first.
These depend on the tyres' lateral forces Ff and Fr which, in turn, 
depend on the tyres' slip angles αf and αr and on cf cr.

The slip angles are:
    αf = δ - atan((v + l_f * r)/u)
    αr =     atan((v - l_r * r)/u)

which gives lateral forces:
    Ff = cf * αf
    Fr - cr * αr

which can be sued to compute v̇ and ω̇:
    v̇ = (Ff + Fr - uω) / m
    ω̇ = (Ff * l_f - Fr * l_r) / Iz

Okay, so we have:
    u̇, δ̇, v̇, ω̇

so we can compute all the rest
    ẋ = u * cos(θ) + v * sin(θ)
    ẏ = u * sin(θ) + v * cos(θ)
    θ̇ = ω

    v̇ = (Ff + Fr - uω) / m
    ω̇ = (Ff * l_f - Fr * l_r) / Iz
    u̇ = control
    δ̇ = control

# --------------------------------- solution --------------------------------- #
The strategy for the solution is the same as for the kinematics probelm.

Recast everything to the track's space using SF and using the track errors n and ψ.
Note that `ṅ` differs from what we had for the kinematics model since we don't have
the velocity slip angle β but we do have the lateral velocity `v`
with:

    ∂(n, s) == SF * (u * sin(ψ) + v)
    ∂(ψ, s) == SF * ω - κ(s)

Similarly, the scaling factor SF goes from:
    SF == (1 - n * κ(s)) / (u * cos(ψ + β) + eps())

to:
    SF == (1 - n * κ(s)) / (u * cos(ψ) + v + eps())
"""
struct DynamicsProblem <: MTMproblem end

"""
Create a `InfiniteOpt` model given a set of parameters and solve it to 
get the controls for the MTM problem.
"""
function create_and_solve_control(
    problem_type::DynamicsProblem,
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
           # velocities
           options.u_bounds.lower ≤ u ≤ options.u_bounds.upper, Infinite(s)
           options.δ_bounds.lower ≤ δ ≤ options.δ_bounds.upper, Infinite(s)

           # velocities & accelerations
           v, Infinite(s)
           ω, Infinite(s)
        #    v̇, Infinite(s)
        #    ω̇, Infinite(s)
           
           # slip angles
           αf, Infinite(s)
           αr, Infinite(s)

           # lateral forces
           Ff, Infinite(s)
           Fr, Infinite(s)
           
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
   l_r, l_f = bike.l_r, bike.l_f
   cf, cr = bike.cf, bike.cr
   m, Iz = bike.m, bike.Iz

   @constraints(
       model,
       begin
            SF == (1 - n * κ(s)) / (u * cos(ψ) + v + eps())  # time -> space domain conversion factor

            # compute slip angles
            αf == δ - atan((v + ω * l_f)/u)
            αr == δ - atan((v - ω * l_r)/u)

            # compute lateral forces
            Ff == cf * αf
            Fr == cr * αr

            # # compute accelerations
            # v̇ = (Ff + Fr - u * ω)/m
            # ω̇ = (Ff * l_f - Fr * l_r)/Iz

            # set dynmics
            ∂(v, s) == (Ff + Fr - u * ω)/m
            ∂(ω, s) == (Ff * l_f - Fr * l_r)/Iz

            # errors
            ∂(n, s) == SF * (u * sin(ψ) + v)
            ∂(ψ, s) == SF * ω - κ(s)

            # controls
            ∂(u, s) == SF * u̇
            ∂(δ, s) == SF * δ̇

            # time
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
           ω(0) == initial_conditions.ω
           v(0) == 0
           αf(0) == 0
           αr(0) == 0
           Ff(0) == 0
           Fr(0) == 0

           # final conditions
           u(track.S_f) == final_conditions.u
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
