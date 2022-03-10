module bicycle
import Parameters: @with_kw
import MyterialColors: blue_grey_darker, blue_grey, cyan_dark

export Bicycle, State

"""
Stores immutable geometric properties of the bike.
Also stores parameters for drawing
"""
@with_kw struct Bicycle
    L::Number   # total lengt
    l::Number   # bike rear wheel to CoM length

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
    x::Number = 0
    y::Number = 0
    θ::Number = 0
    δ::Number = 0
    β::Number = 0
    v::Number = 0
    ω::Number = 0

    n::Number = 0
    ψ::Number = 0
end

end
