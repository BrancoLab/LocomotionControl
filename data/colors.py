from myterial import (
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
)

# variables

speed = amber
angular_velocity = indigo_light
global_coord = green

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
