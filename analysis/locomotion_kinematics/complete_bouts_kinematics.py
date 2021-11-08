# # %%
# import sys

# sys.path.append("./")
# sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
# sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

# import matplotlib.pyplot as plt
# import numpy as np

# from myterial import blue_darker, pink_darker

# from analysis.load import load_complete_bouts
# from data.data_utils import register_in_time, mean_and_std
# from kinematics import track
# import draw

# """
#     Plots complete bouts through the arena,
#     but with the linearized track also
# """

# # get linearized track
# (
#     left_line,
#     center_line,
#     right_line,
#     left_to_track,
#     center_to_track,
#     right_to_track,
#     control_points,
# ) = track.extract_track_from_image(
#     points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
# )


# # load and clean complete bouts
# bouts = load_complete_bouts(
#     keep=1, duration_max=6, linearize_to=center_line, window=10
# )

# # %%
