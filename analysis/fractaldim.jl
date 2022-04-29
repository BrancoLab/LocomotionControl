
module fractaldim
using Plots
using GLM: lm, coef

export estimate_dim

# ---------------------------------------------------------------------------- #
#                                    square                                    #
# ---------------------------------------------------------------------------- #
"""
A single square
"""
struct Square
    x0::Number
    y0::Number
    x1::Number
    y1::Number
    shape::Shape
end

function Square(x0, y0, x1, y1)
    w, h = x1 - x0, y1 - y0
    shape = Shape(x0 .+ [0,w,w,0], y0 .+ [0,0,h,h])
    return Square(x0, y0, x1, y1, shape)
end


"""
Check if any point (x̂, ŷ) ∈ (x,y) is contained by the square.

Usage, give two vectors `x``, `y` and a square `s`:
    
    (x,y) ⊂ s

"""
function ⊂(xy::Tuple{Vector{Float64}, Vector{Float64}}, square::Square)
    x, y = xy
    x0, x1, y0, y1 = square.x0, square.x1, square.y0, square.y1
    return any((x0 .≤ x .≤ x1) .&& (y0 .≤ y .≤ y1))
end



# ---------------------------------------------------------------------------- #
#                                      BOX                                     #
# ---------------------------------------------------------------------------- #
"""
    Box

A Box spanning a region of R² with squares of side Δx.
"""
struct Box
    xmin::Number
    xmax::Number
    ymin::Number
    ymax::Number
    Δx::Number
    squares::Vector{Square}
end

function Box(x, y; Δx=1) 
    xmin = min(x...) - 2*Δx
    xmax = max(x...) + 2*Δx
    ymin = min(y...) - 2*Δx
    ymax = max(y...) + 2*Δx

    squares::Vector{Square} = []

    for x0 in xmin:Δx:xmax
        for y0 in ymin:Δx:ymax
            push!(squares, Square(x0, y0, x0 + Δx, y0 + Δx))
        end
    end

    Box(xmin, xmax, ymin, ymax, Δx, squares)
end

function Box(xmin, xmax, ymin, ymax; Δx=1)
    squares::Vector{Square} = []

    for x0 in xmin:Δx:xmax
        for y0 in ymin:Δx:ymax
            push!(squares, Square(x0, y0, x0 + Δx, y0 + Δx))
        end
    end

    Box(xmin, xmax, ymin, ymax, Δx, squares)
end

# --------------------------------- counting --------------------------------- #
"""
Return the fraction of squares of `box` that are occupied by `xy`.
"""
function fracsquares(xy, box)
    frac = 0
    for square in box.squares
        if xy ⊂ square
            frac += 1
        end
    end
    return frac / length(box.squares)
end

"""
Count the number of occupied squares
"""
function nsquares(xy, box)
    n = 0
    for square in box.squares
        if xy ⊂ square
            n += 1
        end
    end
    return n
end


# --------------------------------- DIMENSION -------------------------------- #
function slope(x, y)
    X = hcat(x, ones(length(x)))
    model = lm(X, y)
    return coef(model)[1]
end

function estimate_dim(x, y, resolutions::Vector{Float64})::Float64
    xmin = min(x...)
    xmax = max(x...)
    ymin = min(y...)
    ymax = max(y...)

    boxes::Vector{Box} = map(res -> Box(xmin, xmax, ymin, ymax; Δx=res), resolutions)
    return estimate_dim(x, y, resolutions, boxes)

end

function estimate_dim(x, y, resolutions::Vector{Float64}, boxes::Vector{Box})::Float64
    N::Vector{Int64} = []
    for box in boxes
        push!(N, nsquares((x, y), box))
    end

    D = abs(slope(
        log.( 1. /resolutions)', log.(N)
    ))
    return D
end


# --------------------------------- plotting --------------------------------- #
function draw_box!(box::Box)
    vline!(box.xmin:box.Δx:box.xmax, lw=1, color="black", alpha=.5, label=nothing)
    hline!(box.ymin:box.Δx:box.ymax, lw=1, color="black", alpha=.5, label=nothing)
    plot!(; xlim=[box.xmin, box.xmax], ylim=[box.ymin, box.ymax+box.Δx])
end


function draw_occupied_squares!(xy, box)
    for square in box.squares
        if xy ⊂ square
            plot!(
                square.shape, color="red", opacity=.5, label=nothing
            )
        end
    end
end


function make_animation(x, y, resolutions::Vector{Float64}; savepath="boxes.gif")
    anim = @animate for i in 1:length(resolutions)
        box = Box(x, y; Δx=resolutions[i])
    
        p = plot(x, y;  lw=4, color="black", label="curve", grid=false)
        draw_box!(box)
        draw_occupied_squares!((x, y), box)
        plot!(;xlim=[-5, 15], ylim=[-3, 3])
    
    
        n = nsquares((x, y), box)
        d = log(n) / log(1/resolutions[i])
        title!("Estimated dimension: $(round(d; digits=2))")
    end
    
    return gif(anim, "test.gif", fps=5)
end
end
