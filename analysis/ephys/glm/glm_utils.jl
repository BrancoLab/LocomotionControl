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
    variables = Term.([
        :s, :v, :v_squared, :dv_250ms, :dv_500ms, :dv_1000ms, :omega, :omega_squared, :domega_250ms, :domega_500ms, :domega_1000ms, :curv_0cm, :curv_10cm, :curv_20cm, :curv_30cm
    ])

    # have a formula missing each individual predictor
    formulas = Dict{String, Any}(
        "missing_$(variables[i])" => (Y ~ sum(variables[1:end .!= i])) for i = 1:length(variables)
    )

    # have a complete formula
    formulas["complete"] = Y  ~ sum(variables)

    # have a formula missing each class of predictors
    formulas["missing_speed_class"] = @formula(p_spike ~ s + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms + curv_0cm + curv_10cm + curv_20cm + curv_30cm)
    formulas["missing_curv_class"] = @formula(p_spike ~ s + v + v_squared + dv_250ms + dv_500ms + dv_1000ms + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms)
    formulas["missing_omega_class"] = @formula(p_spike ~ s + v + v_squared + dv_250ms + dv_500ms + dv_1000ms + curv_0cm + curv_10cm + curv_20cm + curv_30cm)
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
    coefficients = DataFrame(CSV.File(joinpath(
        unit["folder"], "coefficients", "$(best)_$(best_fold).csv"
    )))
    β = coefficients[:, "Coef."]
    
    F = generate_formulas()[best]

    return FittedModel(
        best, F, best_fold, coefficients, β, unit["folder"]
    )
end

""" load best model in a given class """
function load_fitted(unit::Dict, class::String, formulas)
    formulas = something(nothing, generate_formulas())
    correlations = DataFrame(CSV.File(
        joinpath(unit["folder"], "correlations.csv")
    ))
    best_fold = argmax(correlations[:, class]) - 1
    
    # load coefficients
    coefficients = DataFrame(CSV.File(joinpath(
        unit["folder"], "coefficients", "$(class)_$(best_fold).csv"
    )))
    β = coefficients[:, "Coef."]
    
    F = formulas[class]

    return FittedModel(
        class, F, best_fold, coefficients, β, unit["folder"]
    )
end



"""
Load a unit's .parquet dataset
"""
function load_data(unit::Dict)::DataFrame 
    data = DataFrame(read_parquet(unit["unit_data"]))
    data.fold = shuffle!((1:nrow(data)) .% 5)  # 5x k-fold 
    return data
end

""" load dataset for a unit's shuffled data """
function load_data(unit::Dict, shuffle::Int)::DataFrame 
    path = joinpath(unit["folder"], "shuffles", "shuffle_$(shuffle).parquet")
    data = DataFrame(read_parquet(path))
    data.fold = shuffle!((1:nrow(data)) .% 5)  # 5x k-fold 
    return data
end


""" load dataset for a fitted unit. No k-fold """
function load_data(unit::FittedModel)
    path = joinpath(unit.folder, "data.parquet")
    data = DataFrame(read_parquet(path))
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
    (train=view(df, df.fold .!= fold, :),
     test=view(df, df.fold .== fold, :))



# ---------------------------------------------------------------------------- #
#                                    PREDICT                                   #
# ---------------------------------------------------------------------------- #
function predict(model::FittedModel, x::Matrix)
    η = x * model.β
    ŷ = linkinv.(LogitLink(), η)
    return ŷ
end

function predict(model::FittedModel, data::Union{DataFrame, SubDataFrame})
    # get entries from dataset
    x = Matrix(data[:, collect(getfield.(model.formula.rhs, :sym))])

    # add intercept
    x = hcat(ones(Float64, size(x, 1)), x)

    # predict
    return predict(model, x)
end