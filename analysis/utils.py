def check_two_conf_equal(c1, c2):
    for key, val in c1.items():
        if c2[key] != val:
            return False

    return True
