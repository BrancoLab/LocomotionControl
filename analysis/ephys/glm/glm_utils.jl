using Term: install_term_logger, install_term_repr, install_term_stacktrace
using GLM
using Parquet
using CSV
using StatsBase
using DataFrames
using Random
using YAML
import Term.Repr: @with_repr, termshow
import GLM: linkinv

install_term_stacktrace()
install_term_repr()
install_term_logger()


base_folder = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\ephys\\GLM"
metadata = YAML.load_file(joinpath(base_folder, "metadata.yaml"))



# --------------------------------- formulas --------------------------------- #
function generate_formulas()::Dict
    Y = Term(:p_spike)
    variables =
        Term.([
            :s,
            :apex_distance,
            :curv_5cm,
            :curv_15cm,
            :curv_25cm,
            :v,
            :v_squared,
            :dv_250ms,
            :dv_500ms,
            :dv_1000ms,
            :omega,
            :omega_squared,
            :domega_250ms,
            :domega_500ms,
            :domega_1000ms,
        ])

    # # have a formula missing each individual predictor
    # formulas = Dict{String, Any}(
    #     "missing_$(variables[i])" => (Y ~ sum(variables[1:end .!= i])) for i = 1:length(variables)
    # )


    # # have a formula missing each class of predictors
    # formulas["missing_speed_class"] = @formula(p_spike ~ s + apex_distance + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms + curv_5cm + curv_15cm + curv_25cm)
    # formulas["missing_Δspeed_class"] = @formula(p_spike ~ s + apex_distance + v + v_squared + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms + curv_5cm + curv_15cm + curv_25cm)
    # formulas["missing_curv_class"] = @formula(p_spike ~ s + apex_distance + v + v_squared + dv_250ms + dv_500ms + dv_1000ms + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms)
    # formulas["missing_omega_class"] = @formula(p_spike ~ s + apex_distance + v + v_squared + dv_250ms + dv_500ms + dv_1000ms + curv_5cm + curv_15cm + curv_25cm)
    # formulas["missing_Δomega_class"] = @formula(p_spike ~ s + apex_distance + v + v_squared + dv_250ms + dv_500ms + dv_1000ms + omega + omega_squared + curv_5cm + curv_15cm + curv_25cm)


    # make formulas with a single predictor
    formulas = Dict{String,Any}(
        "with_$(variables[i])" => (Y ~ variables[i]) for i = 1:length(variables)
    )

    # make formulas with classes of predictors
    formulas["with_speed_class"] = @formula(p_spike ~ v + v_squared)
    formulas["with_Δspeed_class"] = @formula(p_spike ~ dv_250ms + dv_500ms + dv_1000ms)
    formulas["with_curv_class"] = @formula(p_spike ~ curv_5cm + curv_15cm + curv_25cm)
    formulas["with_omega_class"] = @formula(p_spike ~ omega + omega_squared)
    formulas["with_Δomega_class"] =
        @formula(p_spike ~ domega_250ms + domega_500ms + domega_1000ms)


    # have a complete formula
    formulas["complete"] = Y ~ sum(variables)
    return formulas
end




# ---------------------------------------------------------------------------- #
#                                     LOAD                                     #
# ---------------------------------------------------------------------------- #

@with_repr struct FittedModel
    name::String
    formula::FormulaTerm
    fold::Int
    coefficients::DataFrame  # coefficients summary
    β::Vector   # coefficients values
    folder::String
end


""" Load highest correlation model """
function load_fitted(unit::Dict)
    best = unit["best_formula"]
    best_fold = unit["best_fold"] - 1

    # load coefficients
    coefficients = DataFrame(
        CSV.File(joinpath(unit["folder"], "coefficients", "$(best)_$(best_fold).csv")),
    )
    β = coefficients[:, "Coef."]

    F = generate_formulas()[best]

    return FittedModel(best, F, best_fold, coefficients, β, unit["folder"])
end

""" load best model in a given class """
function load_fitted(unit::Dict, class::String, formulas = nothing)
    formulas = something(nothing, generate_formulas())
    correlations = DataFrame(CSV.File(joinpath(unit["folder"], "correlations.csv")))
    best_fold = argmax(correlations[:, class]) - 1

    # load coefficients
    coefficients = DataFrame(
        CSV.File(joinpath(unit["folder"], "coefficients", "$(class)_$(best_fold).csv")),
    )
    β = coefficients[:, "Coef."]

    F = formulas[class]

    return FittedModel(class, F, best_fold, coefficients, β, unit["folder"])
end


""" load a model in a given shuffle """
function load_fitted(unit::Dict, shuffle::Int, formulas = nothing)
    formulas = something(formulas, generate_formulas())
    F = formulas["complete"]


    # load coefficients
    coefficients = DataFrame(
        CSV.File(joinpath(unit["folder"], "coefficients", "shuffle_$(shuffle)_fold_1.csv")),
    )
    β = coefficients[:, "Coef."]

    return FittedModel("shuffle_$shuffle", F, 1, coefficients, β, unit["folder"])
end



"""
Load a unit's .parquet dataset
"""
function load_data(unit::Dict)::DataFrame
    data = DataFrame(read_parquet(unit["unit_data"]))
    data.p_spike = clamp.(data.p_spike, 0, 1)
    data.fold = shuffle!((1:nrow(data)) .% 5)  # 5x k-fold 
    return data
end

""" load dataset for a unit's shuffled data """
function load_data(unit::Dict, shuffle::Int)::DataFrame
    path = joinpath(unit["folder"], "shuffles", "shuffle_$(shuffle).parquet")
    data = DataFrame(read_parquet(path))
    data.p_spike = clamp.(data.p_spike, 0, 1)
    data.fold = shuffle!((1:nrow(data)) .% 5)  # 5x k-fold 
    return data
end


""" load dataset for a fitted unit. No k-fold """
function load_data(unit::FittedModel)
    path = joinpath(unit.folder, "data.parquet")
    data = DataFrame(read_parquet(path))
    data.p_spike = clamp.(data.p_spike, 0, 1)
    return data
end


"""
    get_fold_data(df::DataFrame, fold::Int)

Get training/test sets for a fold of the data.

#### Usage
```julia
    trainset, testset = get_fold_data(data, 1)
```

Originally from [here](www.juliabloggers.com/dataframes-jl-training-implementing-cross-validation-2)
"""
get_fold_data(df, fold) =
    (train = view(df, df.fold .!= fold, :), test = view(df, df.fold .== fold, :))



# ---------------------------------------------------------------------------- #
#                                    PREDICT                                   #
# ---------------------------------------------------------------------------- #
function predict(model::FittedModel, x::Matrix)
    η = x * model.β
    ŷ = linkinv.(LogitLink(), η)
    return ŷ
end

function predict(model::FittedModel, data::Union{DataFrame,SubDataFrame})
    # get entries from dataset
    if !isa(model.formula.rhs, Term)
        x = Matrix(data[:, collect(getfield.(model.formula.rhs, :sym))])
    else
        x = Vector(data[:, model.formula.rhs.sym])
    end

    # add intercept
    x = hcat(ones(Float64, size(x, 1)), x)

    # predict
    return predict(model, x)
end
