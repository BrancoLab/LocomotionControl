# import jcontrol: FULLTRACK

# import JSONTables: objecttable, jsontable
# using DataFrames


# dd = Dict(
#     "X" => FULLTRACK.X,
#     "Y" => FULLTRACK.Y,
#     "curvature" => FULLTRACK.curvature,   # κ value at each waypoint
#     "S" => FULLTRACK.S,
#     "θ" => FULLTRACK.θ,
# )

# df = DataFrame(dd)
# # @info objecttable(df)
# open("./track.json", "w") do f
#     write(f, objecttable(df))
# end
