"""
Saving/loading functionality
"""
module io
    using Glob
    import JSON
    using DataFrames: DataFrame

    export PATHS, load_trials


    PATHS = Dict(
        "exp_data_fodler" => "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/saved_data",
    )


    humansort(ls) = sort(ls, by=x->parse(Int, match(r"[0-9].*", x).match))

    """
    Load individual bouts metadata and tracking data from individual JSON files.
    Return a DataFrame will all bouts sorted by duration.

    If method==:efficient only a subset of the keys are kept
    If keep_n isa number: only the first keep_n bouts (sorted by duration) are kept
    """
    function load_trials(; keep_n::Union{Nothing, Int}=nothing, method::Symbol=:efficient)::DataFrame
        files::Vector{String} = []
        try
            files = glob("*_bout.json", io.PATHS["exp_data_fodler"])
        catch
            @warn "Could not load tracking data. Perhaps you don't have the right path to the data?"
            return nothing
        end

        # initialize empty dataframe
        f1 = JSON.parsefile(files[1])
        dtype(x) = x isa AbstractVector ? Vector{typeof(x[1])} : typeof(x)
        
        filtervals(d) = values(d)
        if method != :efficient
            # keep all keys
            KEYS = collect(keys(f1))
        else
            KEYS = ["body_x", "body_y", "complete", "direction", "duration", "mouse_id", "name", "start_frame", "end_frame", "gcoord"]
            filtervals(d::Dict) = map((x)->get(d, x, nothing), KEYS)
        end
        data = Dict(k=>dtype(v)[] for (k,v) in zip(KEYS, filtervals(f1)))

        # collect data
        cleanvec(x) = x isa AbstractVector ? convert(Vector{typeof(x[1])}, x) : x
        for fl in files
            # open file
            contents = JSON.parsefile(fl)
            for (k,v) in zip(KEYS, cleanvec.(filtervals(contents)))

                # fix data misalignment
                if contains(k, "_x")
                    v .+= .5
                elseif contains(k, "_y")
                    v .-= 1
                end
                push!(data[k], v)
            end
        end

        # sort by duration and shortem
        data = sort!(DataFrame(data), :duration)
        data = isnothing(keep_n) ? data : first(data, keep_n)

    end

end
