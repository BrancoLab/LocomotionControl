from __future__ import annotations

import numpy as np
from typing import Union, Tuple, List

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
        if self.single_vec:
            return f"Vector @ ({self.x}, {self.y})"
        else:
            return f"Array of {len(self.x)} vectors."

    def __len__(self):
        if not self.single_vec:
            return len(self.x)
        else:
            return 1

    def __getitem__(self, item) -> Vector:
        if self.single_vec:
            raise IndexError("Cannot index vector")
        else:
            return Vector(self.x[item], self.y[item])

    def to_polar(self) -> Tuple[Union[np.ndarray, float]]:
        rho = np.hypot(self.x, self.y)
        phi = np.degrees(np.arctan2(self.y, self.x))
        return rho, phi

    @classmethod
    def from_list(cls, vecs: List[Vector]) -> Vector:
        return Vector([v.x for v in vecs], [v.y for v in vecs])

    @property
    def single_vec(self):
        return isinstance(self.x, float)

    @property
    def angle(self) -> Union[np.ndarray, float]:
        return np.degrees(np.arctan2(self.y, self.x))

    @property
    def magnitude(self) -> Union[np.ndarray, float]:
        if self.single_vec:
            return np.sqrt(self.x ** 2 + self.y ** 2)
        else:
            # compute norm of each vector
            vec = np.vstack([self.x, self.y]).T
            return np.apply_along_axis(np.linalg.norm, 1, vec)

    def dot(
        self, other: Vector, norm: bool = True
    ) -> Union[float, np.ndarray]:
        if len(self) != len(other):
            raise ValueError("only vectors of same length can be dotted")

        # compute dot product
        if self.single_vec:
            _dot = np.dot([self.x, self.y], [other.x, other.y])
        else:
            _dot = np.array(
                [
                    np.dot([self.x[i], self.y[i]], [other.x[i], other.y[i]])
                    for i in range(len(self))
                ]
            )

        # return normalize dot
        if norm:
            return _dot / other.magnitude
        else:
            return _dot
