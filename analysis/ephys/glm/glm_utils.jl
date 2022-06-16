using Term: install_term_logger, install_term_repr, install_term_stacktrace
using GLM
using Parquet
using CSV
using StatsBase
using DataFrames
using Random

install_term_stacktrace()
install_term_repr()
install_term_logger()


base_folder = raw"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys\GLM"
metadata = YAML.load_file(joinpath(base_folder, "metadata.yaml"))


"""
Load a unit's .parquet dataset
"""
function load_data(unit::Dict)::DataFrame 
    data = DataFrame(read_parquet(unit["unit_data"]));
    data.fold = shuffle!((1:nrow(data)) .% 5)  # 5x k-fold 
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



# --------------------------------- formulas --------------------------------- #
function generate_formulas()::Dict
    Y = Term(:p_spike)
    variables = Term.([
        :v, :v_squared, :dv_250ms, :dv_500ms, :dv_1000ms, :omega, :omega_squared, :domega_250ms, :domega_500ms, :domega_1000ms, :curv_0cm, :curv_10cm, :curv_20cm, :curv_30cm
    ])

    # have a formula missing each individual predictor
    formulas = Dict{String, Any}(
        "missing_$(variables[i])" => (Y ~ sum(variables[1:end .!= i])) for i = 1:length(variables)
    )

    # have a complete formula
    formulas["complete"] = Y  ~ sum(variables)

    # have a formula missing each class of predictors
    formulas["missing_speed_class"] = @formula(p_spike ~ omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms + curv_0cm + curv_10cm + curv_20cm + curv_30cm)
    formulas["missing_curv_class"] = @formula(p_spike ~ v + v_squared + dv_250ms + dv_500ms + dv_1000ms + omega + omega_squared + domega_250ms + domega_500ms + domega_1000ms)
    formulas["missing_omega_class"] = @formula(p_spike ~ v + v_squared + dv_250ms + dv_500ms + dv_1000ms + curv_0cm + curv_10cm + curv_20cm + curv_30cm)
    return formulas
end




# ---------------------------------------------------------------------------- #
#                                     LOAD                                     #
# ---------------------------------------------------------------------------- #

struct FittedModel
    name::String
    formula::FormulaTerm
    fold::Int
    coefficients::DataFrame  # coefficients summary
    β::Vector   # coefficients values
end

function load_fitted(unit::Dict)
    best = unit["best_formula"]
    best_fold = unit["best_fold"] - 1
    
    # load coefficients
    unit_folder = splitdir(doing["unit_data"])[1]
    coefficients = DataFrame(CSV.File(joinpath(
        unit_folder, "coefficients", "$(best)_$(best_fold).csv"
    )))
    β = coefficients[:, "Coef."]
    
    F = formulas[best]

    return FittedModel(
        best, F, best_fold, coefficients, β
    )
end


# ---------------------------------------------------------------------------- #
#                                    PREDICT                                   #
# ---------------------------------------------------------------------------- #
function predict(model::FittedModel, x::Matrix)
    η = x * model.β
    ŷ = linkinv.(LogitLink(), η)
    return ŷ
end

function predict(model::FittedModel, data::DataFrame)
    # get entries from dataset
    x = Matrix(data[:, collect(getfield.(model.formula.rhs, :sym))])

    # add intercept
    x = hcat(ones(Float64, size(x, 1)), x)

    # predict
    return predict(model, x)
end