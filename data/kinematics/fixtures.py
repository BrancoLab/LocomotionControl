from myterial import (
    salmon,
    salmon_darker,
    indigo,
    indigo_darker,
    red,
    red_darker,
    red_light,
)

BODY_PARTS_NAMES = (
    "snout",
    "right_ear",
    "right_fl",
    "right_hl",
    "tail_base",
    "left_hl",
    "left_fl",
    "left_ear",
    "body",
)
HEAD_NAMES = ("snout", "right_ear", "body", "left_ear")
PAWS_NAMES = ("right_fl", "right_hl", "left_hl", "left_fl")
BODY_NAMES = ("right_fl", "right_hl", "tail_base", "left_hl", "left_fl")
BODY_AXIS_NAMES = ("snout", "body", "tail_base")

BODY_PARTS_COLORS = dict(
    left_hl=salmon,
    left_fl=salmon_darker,
    left_ear=red,
    snout=red_darker,
    body=red,
    tail_base=red_light,
    right_hl=indigo,
    right_fl=indigo_darker,
    right_ear=red,
    left_paws="k",
    right_paws="k",
)

PAWS_COLORS = dict(
    left_fl=salmon_darker,
    right_fl=indigo_darker,
    left_hl=salmon,
    right_hl=indigo,
)
