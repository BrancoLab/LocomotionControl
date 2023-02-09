using Term.Progress
using DataFrames
import JSONTables: objecttable, jsontable

import jcontrol: FULLTRACK
import jcontrol.io: load_trials
import jcontrol.trial: Trial

"""
Save each "raw" trial from python to a json file

"""
# folder with json files exported from python's database
TRIALS_FOLDER = raw"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior\saved_data"
# TRIALS_FOLDER = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\ephys\\locomotion_bouts\\saved_data"

# folder where processed/cached julia trials will be stored
CACHE_FOLDER = raw"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\behavior\jl_trials_cache"
# CACHE_FOLDER = "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\ephys\\locomotion_bouts\\processed"

@info "Caching data" TRIALS_FOLDER CACHE_FOLDER

# load trials
trials = load_trials(; folder=TRIALS_FOLDER)

# cache trials
pbar = ProgressBar()
job = addjob!(pbar; N=size(trials, 1))

metadata_keys = (
    "mouse_id", "name", "direction", "duration", "start_frame", "end_frame", "complete"
)
paws_keys = (
    "left_fl_x",
    "left_fl_y",
    "right_fl_x",
    "right_fl_y",
    "left_hl_x",
    "left_hl_y",
    "right_hl_x",
    "right_hl_y",
)
with(pbar) do
    for (i, rawtrial) in enumerate(eachrow(trials))
        trial = Trial(rawtrial, FULLTRACK; fixstart=false)

        trialdict = Dict(
            "x" => trial.x,
            "y" => trial.y,
            "s" => trial.s,
            "theta" => trial.θ,
            "omega" => trial.ω,
            "speed" => trial.speed,
            "u" => trial.u,
            "v" => trial.v,
            "duration" => trial.duration,
        )
        for k in metadata_keys
            trialdict[k] = rawtrial[k]
        end
        for k in paws_keys
            trialdict[k] = rawtrial[k]
        end
        @assert length(trial.x) == length(rawtrial["body_x"]) "$(length(trial.x)) $(length(rawtrial["body_x"]))"

        df = DataFrame(trialdict)

        open(joinpath(CACHE_FOLDER, "trial_$(i).json"), "w") do f
            write(f, objecttable(df))
        end

        update!(job)
    end
end

# open(joinpath(CACHE_FOLDER, "trial_1.json")) do f
#     data = read(f)
#     @time jsontable(data)
#     print(jsontable(data).x)
# end
