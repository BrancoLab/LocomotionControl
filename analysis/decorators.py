from functools import wraps
from loguru import logger
import numpy as np
import pandas as pd


def log_function_call(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        args_names = [a if not isinstance(a, (np.ndarray, pd.DataFrame, pd.Series)) else type(a) for a in args]
        logger.debug(f'Calling {f.__name__} with arguments: {args_names}, {kwargs}')
        return f(*args, **kwargs)
    return wrapper