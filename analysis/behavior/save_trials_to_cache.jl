import Term.progress: track as pbar
using DataFrames
import JSONTables: objecttable, jsontable


import jcontrol: FULLTRACK
import jcontrol.io: load_trials
import jcontrol.trial: Trial


"""
Save each "raw" trial from python to a json file

"""

# trials = @time load_trials()
folder = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\behavior\\jl_trials_cache"


# for (i, trial) in pbar(enumerate(eachrow(trials)); redirectstdout=false)
#     trial = Trial(trial, FULLTRACK)

#     trialdict = Dict(
#         "x" => trial.x,
#         "y" => trial.y,
#         "s" => trial.s,
#         "θ" => trial.θ,
#         "ω" => trial.ω,
#         "u" => trial.u,
#         "duration" => trial.duration,
#     )

#     df = DataFrame(trialdict)
#     # @info objecttable(df)
#     open(joinpath(folder, "trial_$(i).json"), "w") do f
#         write(f, objecttable(df))
#     end
# end


# open(joinpath(folder, "trial_1.json")) do f
#     data = read(f)
#     @time jsontable(data)
#     print(jsontable(data).x)
# end

@time Trial(joinpath(folder, "trial_1.json"))

