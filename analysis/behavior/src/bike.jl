module bicycle
import Parameters: @with_kw
import MyterialColors: blue_grey_darker, blue_grey, cyan_dark

export Bicycle, State

"""
Stores immutable geometric properties of the bike.
Also stores parameters for drawing.
"""
@with_kw struct Bicycle

    # geometry
    l_f::Number     # distance from COM to front wheel
    l_r::Number     # distance from rear wheel to COM
    width::Number   # 'width' of bike. Used to staty within track

    # dynamics
    m = 25  # grams
    Iz = 2  # moment of angular inertia
    cf = 1  # corenring stiffness
    cr = 1  # cornering stiffnes

    # for drawing
    wheel_length = 0.8
    wheel_lw = 16
    wheel_color = blue_grey_darker
    front_wheel_color = cyan_dark
    body_lw = 8
    body_color = blue_grey
end

"""
Represents the state of the bicycle model at a moment in time.
Can be used to pass initial and final condistions to the control
model.
"""
@with_kw struct State
    x::Number = 0  # position
    y::Number = 0
    θ::Number = 0  # orientation
    δ::Number = 0  # steering angle
    ω::Number = 0
    u::Number = 0  # velocity  | fpr DynamicsProblem its the longitudinal velocity component

    # KinematicsProblem only
    β::Number = 0  # slip angle

    # DynamicsProblem only
    v::Number = 0  # lateral velocity

    # track errors 
    n::Number = 0
    ψ::Number = 0
end

end
