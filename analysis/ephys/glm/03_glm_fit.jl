using GLM
using DataFrames
using Random
using Parquet
import YAML

include(raw"C:\Users\Federico\Documents\GitHub\pysical_locomotion\analysis\ephys\glm\glm_utils.jl")




function update_metadata(metadata, key, correlations)
    # save correlations
    correlations = DataFrame(correlations)
    CSV.write(joinpath(metadata[key]["folder"], "correlations.csv"), correlations)

    # get the average cross validated correlation of each formula
    mean_corr = mean.(eachcol(correlations))
    best = collect(keys(formulas))[argmax(mean_corr)]

    # update metadata
    metadata[key]["best_formula"] = best
    metadata[key]["best_fold"] = argmax(correlations[:, best])
    metadata[key]["best_corr"] = max(mean_corr...)
    metadata[key]["glm_fitted"] = true

    YAML.write_file(joinpath(base_folder, "metadata.yaml"), metadata)
end

# ---------------------------------------------------------------------------- #
#                                      FIT                                     #
# ---------------------------------------------------------------------------- #
function fit_model(doing::Dict, data::DataFrame, formulas::Dict)
    coefficients_folder = joinpath(doing["folder"], "coefficients")
    ispath(coefficients_folder) || mkdir(coefficients_folder)

    predictions_folder = joinpath(doing["folder"], "predictions")
    ispath(predictions_folder) || mkdir(predictions_folder)

    # keep track of the Parson's correlation of each model
    correlations = Dict{String, Vector}(name => [] for name in keys(formulas))

    # fit each model on all folders
    for (name, formula) in formulas
        for fold in 0:4
            # fit model
            train, test = get_fold_data(data, fold)
            model = glm(formula, train, Binomial(), LogitLink())
    
            # save coefficients
            coefficients = coeftable(model) |> DataFrame
            CSV.write(joinpath(coefficients_folder, "$(name)_$(fold).csv"), coefficients)
    
            # save predictions
            y = test[:,:p_spike]
            ŷ = predict(model, test)
            CSV.write(joinpath(predictions_folder, "$(name)_$(fold).csv"), DataFrame(
                        Dict(:y=>y, :ŷ=>ŷ))
            )
    
            # get Pearson's correlation coefficient
            push!(correlations[name], cor(y, ŷ))
    end
    return correlations
end


# ---------------------------------------------------------------------------- #
#                                    EXECUTE                                   #
# ---------------------------------------------------------------------------- #
function main()
    formulas = generate_formulas()

    for (key, doing) in metadata
        doing["glm_fitted"] && continue

        # load data
        data = load_data(doing)

        # fit & save
        correlations = fit_model(doing, data, formulas)
        update_metadata(metadata, key, correlations)

        break
    end
    
end

@time main()
