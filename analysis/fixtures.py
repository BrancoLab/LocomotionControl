from collections import namedtuple


_bps = namedtuple('bps', 'body, left_fl, right_fl, right_hl, left_hl, neck, tail_base')
BPS = _bps('body', 'left_fl', 'right_fl', 'right_hl', 'left_hl', 'neck', 'tail_base')


_paws = namedtuple('paws', 'left_fl, right_fl, right_hl, left_hl')
PAWS = _paws('left_fl', 'right_fl', 'right_hl', 'left_hl')
