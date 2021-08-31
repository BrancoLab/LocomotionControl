from myterial import pink_light, blue_light, blue_grey_dark

# body parts colors


# hairpin moving direction
outbound = pink_light
inbound = blue_light
bout_direction_colors = {
    1: inbound,
    -1: outbound,
    0: blue_grey_dark,
    "inbound": inbound,
    "outbound": outbound,
    "none": blue_grey_dark,
}
