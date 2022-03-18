module bicycle
import Parameters: @with_kw
import MyterialColors: blue_grey_darker, blue_grey, cyan_dark

export Bicycle, State

"""
Stores immutable geometric properties of the bike.
Also stores parameters for drawing.
"""
struct Bicycle

    # geometry
    l_f::Number     # distance from COM to front wheel | cm
    l_r::Number     # distance from rear wheel to COM | cm
    L::Number       # total length | cm
    width::Number   # 'width' of bike. Used to staty within track | cm

    # dynamics
    m::Number       # mass | g
    Iz::Number      # moment of angular inertia | Kg⋅m²
    cf::Number      # corenring stiffness
    cr::Number      # cornering stiffnes

    # for drawing
    wheel_length::Number
    wheel_lw::Number
    wheel_color::String
    front_wheel_color::String
    body_lw::Number
    body_color::String

    function Bicycle(;
                l_f::Number=5,
                l_r::Number=3,
                width::Number=2,
                m_f = 10,
                m_r = 15,
                cf = .1,  # 0.5 ≤ cf ≤ 3.5 generally works
                cr = .1,
        )

        # convert units g->Kg, cm->m
        mfKg = m_f / 100
        mrKg = m_r / 100
        lfM  = l_f / 100
        lrM  = l_r / 100

        # compute moment of angular inertia        
        Iz = mfKg * lfM^2 + mrKg * lrM^2


        return new(
            l_f,
            l_r,
            l_f + l_r, 
            width,
            m_f + m_r,
            Iz,
            cf,
            cr,
            0.8,                # wheel_length
            16,                 # wheel_lw
            blue_grey_darker,   # wheel_color
            cyan_dark,          # front_wheel_color
            8,                  # body_lw
            blue_grey,          # body_color
        )


    end
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
