using Term.progress
using DataFrames
import JSONTables: objecttable, jsontable


import jcontrol: FULLTRACK
import jcontrol.io: load_trials
import jcontrol.trial: Trial


"""
Save each "raw" trial from python to a json file

"""
# folder with json files exported from python's database
TRIALS_FOLDER = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/inbound_bouts"  

# folder where processed/cached julia trials will be stored
CACHE_FOLDER  = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/jl_inbound_trials"  

@info "Caching data" TRIALS_FOLDER CACHE_FOLDER

# load trials
trials = load_trials(; folder=TRIALS_FOLDER)

# cache trials
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

        open(joinpath(CACHE_FOLDER, "trial_$(i).json"), "w") do f
            write(f, objecttable(df))
        end

        update!(job)
        sleep(0.001)
    end
end

# open(joinpath(CACHE_FOLDER, "trial_1.json")) do f
#     data = read(f)
#     @time jsontable(data)
#     print(jsontable(data).x)
# end


