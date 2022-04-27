using Term.progress
using DataFrames
import JSONTables: objecttable, jsontable


import jcontrol: FULLTRACK
import jcontrol.io: load_trials
import jcontrol.trial: Trial


"""
Save each "raw" trial from python to a json file

"""

trials = load_trials()
# folder = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\behavior\\jl_trials_cache"
folder = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/jl_trials_cache"


pbar = ProgressBar()
job = addjob!(pbar; N=size(trials, 1))

with(pbar) do

    for (i, trial) in enumerate(eachrow(trials))
        trial = Trial(trial, FULLTRACK)

        trialdict = Dict(
            "x" => trial.x,
            "y" => trial.y,
            "s" => trial.s,
            "θ" => trial.θ,
            "ω" => trial.ω,
            "speed" => trial.speed,
            "u" => trial.u,
            "v" => trial.v,
            "duration" => trial.duration,
        )

        df = DataFrame(trialdict)
        # @info objecttable(df)
        open(joinpath(folder, "trial_$(i).json"), "w") do f
            write(f, objecttable(df))
        end

        # i > 300 && break
        update!(job)
        sleep(0.001)
    end
end

open(joinpath(folder, "trial_1.json")) do f
    data = read(f)
    @time jsontable(data)
    print(jsontable(data).x)
end


