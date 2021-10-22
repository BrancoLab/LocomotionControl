import numpy as np
from typing import Union


# TODO define vector addition and scalar multiplication operations


class Vector:  # 2D vector
    """
        It can either represent a single vector (x,y are floats) or a 
        list of vecotrs (x, y are 1D numpy arrays)
    """

    def __init__(
        self,
        x: Union[np.ndarray, list, float],
        y: Union[np.ndarray, list, float] = None,
    ):
        if y is None:
            y, x = x[:, 1], x[:, 0]

        self.x = x
        self.y = y

    def __repr__(self):
        if isinstance(self.x, float):
            return f"Vector @ ({self.x}, {self.y})"
        else:
            return f"Array of {len(self.x)} vectors."

    @property
    def angle(self) -> Union[np.ndarray, float]:
        return np.degrees(np.arctan2(self.y, self.x))

    @property
    def magnitude(self) -> Union[np.ndarray, float]:
        if isinstance(self.x, float):
            return np.sqrt(self.x ** 2 + self.y ** 2)
        else:
            # compute norm of each vector
            vec = np.vstack([self.x, self.y]).T
            return np.apply_along_axis(np.linalg.norm, 1, vec)
