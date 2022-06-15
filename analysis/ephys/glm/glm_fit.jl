using GLM
using DataFrames
using Random
using Parquet
import YAML


base_folder = raw"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys\GLM"
metadata = YAML.load_file(joinpath(base_folder, "metadata.yaml"))


doing = collect(values(metadata))[1]

# ---------------------------------------------------------------------------- #
#                                   LOAD DATA                                  #
# ---------------------------------------------------------------------------- #
function load_data(doing::Dict)
    data = DataFrame(read_parquet(doing["unit_data"]));

    # add a column with the fold number of each row (shuffled!)
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


# ---------------------------------------------------------------------------- #
#                                DEFINE FORMULAS                               #
# ---------------------------------------------------------------------------- #
function generate_formulas()
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
#                                      FIT                                     #
# ---------------------------------------------------------------------------- #
function fit_model(data::DataFrame, formulas::Dict)
    for (name, formula) in formulas
        for fold in 1:5
            train, test = get_fold_data(data, fold)
            model = glm(formula, train, Binomial(), LogitLink())
            # print(model)
            # model.summary(test)
            # break
        end
        # break
    end
end


# ---------------------------------------------------------------------------- #
#                                    EXECUTE                                   #
# ---------------------------------------------------------------------------- #
function main()
    data = load_data(doing)
    fit_model(data, generate_formulas())
end

@time main()
