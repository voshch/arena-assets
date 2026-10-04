from __future__ import annotations

import typing


class BoundingBox(tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
    def __new__(cls, sides: typing.Iterable[typing.Iterable[float]]) -> typing.Self:
        pairs = tuple((low, high) for low, high in sides)
        if len(pairs) != 3:
            raise ValueError(f"BoundingBox needs exactly 3 sides, got {len(pairs)}")
        return super().__new__(cls, pairs)

    @property
    def min_x(self) -> float:
        return self[0][0]

    @property
    def max_x(self) -> float:
        return self[0][1]

    @property
    def min_y(self) -> float:
        return self[1][0]

    @property
    def max_y(self) -> float:
        return self[1][1]

    @property
    def min_z(self) -> float:
        return self[2][0]

    @property
    def max_z(self) -> float:
        return self[2][1]

    @property
    def volume(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)

    @classmethod
    def empty(cls) -> BoundingBox:
        return cls(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

    def round(self, ndigits: int = 4) -> BoundingBox:
        return BoundingBox(
            (
                (round(self.min_x, ndigits), round(self.max_x, ndigits)),
                (round(self.min_y, ndigits), round(self.max_y, ndigits)),
                (round(self.min_z, ndigits), round(self.max_z, ndigits)),
            )
        )
