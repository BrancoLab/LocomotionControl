from myterial import (
    purple_light,
    pink_light,
    blue_light,
    blue_grey_dark,
    amber,
    indigo_light,
    green,
    orange,
    salmon,
    indigo_dark,
    blue_dark,
    purple,
    blue,
    pink,
)

# variables

speed = amber
angular_velocity = indigo_light
dmov_velocity = indigo_light
velocity = blue
acceleration = pink
global_coord = green

variables = dict(
    speed=speed,
    angular_velocity=angular_velocity,
    dmov_velocity=dmov_velocity,
    velocity=velocity,
    acceleration=acceleration,
    global_coord=global_coord,
)

# movements
moving = orange
walking = salmon
turning_left = indigo_dark
turning_right = blue_dark
tone_on = purple

movements = dict(
    moving=moving,
    walking=walking,
    turning_left=turning_left,
    turning_right=turning_right,
    tone_on=tone_on,
)


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

# body parts colors
snout = "#C7D3E6"
neck = "#7D95BA"
body = "#5071A5"
tail_base = "#265194"
left_fl = pink_light
left_hl = pink
right_fl = purple_light
right_hl = purple


bodyparts = dict(
    snout=snout,
    neck=neck,
    body=body,
    tail_base=tail_base,
    left_fl=left_fl,
    left_hl=left_hl,
    right_fl=right_fl,
    right_hl=right_hl,
)
