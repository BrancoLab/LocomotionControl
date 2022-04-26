def inbounds(var, low, high):
    return min(high, max(var, low))


def normalize_to_unitbox(v, a, b):
    """
        Normalize a value v in [a, b] to the [-1, 1]
        unit box
    """
    return 2 * (v - a) / (b - a) - 1


def unnormalize(v, a, b):
    """
        Take a value v in [-1, 1] and unnormalize it to
        the original range [a, b]
    """
    return (v + 1) / 2 * (b - a) + a

