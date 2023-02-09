using GLM
using DataFrames
using Random
using Parquet
import YAML
import GLM: coeftable
import GLM
import Term: tprint

include(
    raw"C:\Users\Federico\Documents\GitHub\pysical_locomotion\analysis\ephys\glm\glm_utils.jl",
)

DO_SHUFFLES = true


function wrapup(metadata, key, correlations, shuffled_correlations, formulas)
    # save correlations
    correlations = DataFrame(correlations)
    CSV.write(joinpath(metadata[key]["folder"], "correlations.csv"), correlations)

    # save shuffled correlations
    if DO_SHUFFLES
        shuffled_correlations = DataFrame(shuffled_correlations)
        CSV.write(
            joinpath(metadata[key]["folder"], "shuffled_correlations.csv"),
            shuffled_correlations,
        )
    end

    # get the average cross validated correlation of each formula
    mean_corr = mean.(eachcol(correlations))
    best = collect(keys(formulas))[argmax(mean_corr)]

    # update metadata
    metadata[key]["best_formula"] = best
    metadata[key]["best_fold"] = argmax(correlations[:, best])
    metadata[key]["best_corr"] = max(mean_corr...)
    metadata[key]["glm_fitted"] = true
end

# ---------------------------------------------------------------------------- #
#                                      FIT                                     #
# ---------------------------------------------------------------------------- #

df2mtx(df::SubDataFrame, F::FormulaTerm) = Matrix{Float64}(
    hcat(df[:, collect(getfield.(F.rhs, :sym))], ones(Float64, size(df, 1))),
)

function fit_model(doing::Dict, data::DataFrame, formulas::Dict)
    coefficients_folder = joinpath(doing["folder"], "coefficients")
    ispath(coefficients_folder) || mkdir(coefficients_folder)

    predictions_folder = joinpath(doing["folder"], "predictions")
    ispath(predictions_folder) || mkdir(predictions_folder)

    # keep track of the Parson's correlation of each model
    correlations = Dict{String,Vector}(name => zeros(Float64, 5) for name in keys(formulas))

    # fit each model on all folders
    for (name, formula) in formulas
        for fold = 0:4
            # fit model
            train, test = get_fold_data(data, fold)
            model = glm(formula, train, Binomial(), LogitLink(); maxiter = 100)

            # save coefficients
            coefficients = coeftable(model) |> DataFrame
            CSV.write(joinpath(coefficients_folder, "$(name)_$(fold).csv"), coefficients)

            # save predictions
            y = test[:, :p_spike]
            ŷ = GLM.predict(model, test)
            CSV.write(
                joinpath(predictions_folder, "$(name)_$(fold).csv"),
                DataFrame(:y => y, :ŷ => ŷ),
            )

            # get Pearson's correlation coefficient
            correlations[name][fold+1] = cor(y, ŷ)
        end
    end


    # fit on each shuffled dataset
    shuffled_correlations = Dict{String,Vector}(string(i) => zeros(Float64, 5) for i = 0:99)

    if DO_SHUFFLES
        F = formulas["complete"]
        X_train = map(i -> df2mtx(get_fold_data(data, i)[1], F), 0:4)  # vector of matrices
        X_test = map(i -> df2mtx(get_fold_data(data, i)[2], F), 0:4)

        for i = 0:99
            y = Vector{Float64}(load_data(doing, i).p_spike)  # get p spike for new unit
            for fold = 0:4
                # fit model
                y_train = view(y, data.fold .!= fold)
                y_test = view(y, data.fold .== fold)

                model = GLM.fit(
                    GeneralizedLinearModel,
                    X_train[fold+1],
                    y_train,
                    Binomial(),
                    LogitLink();
                    maxiter = 100,
                )

                # save coefficients
                coefficients = coeftable(model) |> DataFrame
                CSV.write(
                    joinpath(coefficients_folder, "shuffle_$(i)_fold_$(fold).csv"),
                    coefficients,
                )

                # get Pearson's correlation coefficient
                shuffled_correlations[string(i)][fold+1] =
                    cor(y_test, GLM.predict(model, X_test[fold+1]))
            end
        end
    end

    return correlations, shuffled_correlations
end


# ---------------------------------------------------------------------------- #
#                                    EXECUTE                                   #
# ---------------------------------------------------------------------------- #


function main()
    formulas = generate_formulas()

    metadata = YAML.load_file(joinpath(base_folder, "metadata.yaml"))

<<<<<<< HEAD
    to_run = filter(k -> !metadata[k]["glm_fitted"], keys(metadata)) |> collect
    to_run = collect(keys(metadata))
=======
    to_run = filter(
        k -> metadata[k]["glm_fitted"] == false, keys(metadata)
    ) |> collect
>>>>>>> 2066bfa38e94a6a54c389b59139ab53f166c6625

    @info "Fitting GLM on $(length(to_run)) cells on $(Threads.nthreads()) threads"
    @info "This means {bold red}$(length(metadata) - length(to_run)){/bold red} cells have already been fitted"

    done = []
    count = 0
    try
        Threads.@threads for key in to_run
<<<<<<< HEAD
            # for key in to_run
=======
        # for key in to_run
        # [1:2]
>>>>>>> 2066bfa38e94a6a54c389b59139ab53f166c6625
            doing = metadata[key]

            count += 1
            tprint(
                "Doing: '$(doing["recording"])' --> $(doing["unit"]) on thread $(Threads.threadid()) | $(count)/$(length(to_run))\n",
            )

            # load data
            data = load_data(doing)

            # fit
            correlations, shuffled_correlations = fit_model(doing, data, formulas)
            wrapup(metadata, key, correlations, shuffled_correlations, formulas)
            push!(done, key)

<<<<<<< HEAD
            length(done) % 20 == 0 && begin
                @info "Saving metadata ($(length(done)) done so far)"
=======
            length(done) % 5 == 0 && begin
            @info "Saving metadata ($(length(done)) done so far)"    
>>>>>>> 2066bfa38e94a6a54c389b59139ab53f166c6625
                YAML.write_file(joinpath(base_folder, "metadata.yaml"), metadata)

                metadata = YAML.load_file(joinpath(base_folder, "metadata.yaml"))

                to_run_now = filter(
                    k -> metadata[k]["glm_fitted"] == false, keys(metadata)
                ) |> collect
            
                @info "A total of {bold red}$(length(metadata) - length(to_run_now)){/bold red} cells have already been fitted"
            end
        end
    catch err
        @warn err
        rethrow(err)
    end


    # update metadata file
    @info "Updating metadata"
    YAML.write_file(joinpath(base_folder, "metadata.yaml"), metadata)

end

@time main()
