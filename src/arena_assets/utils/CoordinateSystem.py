from __future__ import annotations

import typing

import numpy as np
from scipy.spatial.transform import Rotation as R

Axis = typing.Literal["X+", "X-", "Y+", "Y-", "Z+", "Z-"]


class AxisMap:
    __map: dict[Axis, tuple[int, int, int]] = {
        "X+": (1, 0, 0),
        "X-": (-1, 0, 0),
        "Y+": (0, 1, 0),
        "Y-": (0, -1, 0),
        "Z+": (0, 0, 1),
        "Z-": (0, 0, -1),
    }

    @classmethod
    def vector(cls, axis: Axis) -> tuple[int, int, int]:
        return cls.__map[axis]

    @classmethod
    def axis(cls, vector: typing.Iterable[float]) -> Axis:
        key = tuple(int(round(component)) for component in vector)
        for axis, vec in cls.__map.items():
            if vec == key:
                return axis
        raise ValueError(f"Vector {key} is not a valid axis representation.")


class CoordinateSystem:
    """
    Represents a 3D right-handed coordinate system.

    This class creates an orientation based on a definition of its local axes
    in terms of a standard world coordinate system (+X=right, +Y=up, +Z=forward).
    """

    @classmethod
    def default(cls) -> CoordinateSystem:
        """
        Returns Blender's default world coordinate system (X-right, Z-up, Y-back).
        """
        return cls("X+", "Z+", "Y-")

    def __init__(self, right: Axis, up: Axis, front: Axis):
        try:
            x_dir = AxisMap.vector(right)
            y_dir = AxisMap.vector(up)
            z_dir = AxisMap.vector(front)
        except KeyError as e:
            raise ValueError(f"Invalid axis identifier: {e}") from e

        rotation_matrix = np.column_stack((x_dir, y_dir, z_dir))

        # Validate that the system is right-handed and orthogonal
        if not np.isclose(np.linalg.det(rotation_matrix), 1.0):
            raise ValueError(f"Axes {(right, up, front)} do not form a valid right-handed coordinate system.")

        self._rotation = R.from_matrix(rotation_matrix)

    @property
    def rotation(self) -> R:
        return self._rotation

    def get_transformation_to(self, target_system: CoordinateSystem) -> tuple[float, float, float]:
        """
        Calculates the transformation to align this system with a target system.

        @param target_system: Another CoordinateSystem instance to align to.
        @return: A tuple of radians rotations (x, y, z).
        """
        transform_rot = target_system.rotation * self.rotation.inv()
        angles_rad = transform_rot.as_euler("xyz")
        return tuple(angles_rad)

    def __repr__(self) -> str:
        """
        Provides a string representation of the CoordinateSystem.
        """
        return f"CoordinateSystem({AxisMap.axis(self.rotation.as_matrix()[:, 0])}, {AxisMap.axis(self.rotation.as_matrix()[:, 1])}, {AxisMap.axis(self.rotation.as_matrix()[:, 2])})"


__all__ = ["CoordinateSystem"]
