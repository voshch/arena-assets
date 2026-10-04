import math

import pytest

from arena_assets.utils.CoordinateSystem import CoordinateSystem


def test_identity_transformation():
    blender = CoordinateSystem.default()
    assert blender.get_transformation_to(blender) == pytest.approx((0.0, 0.0, 0.0))


def test_y_up_to_blender():
    y_up = CoordinateSystem("X+", "Y+", "Z+")
    assert y_up.get_transformation_to(CoordinateSystem.default()) == pytest.approx(
        (math.pi / 2, 0.0, 0.0)
    )


def test_left_handed_rejected():
    with pytest.raises(ValueError):
        CoordinateSystem("X+", "Z+", "Y+")


def test_repr():
    assert repr(CoordinateSystem.default()) == "CoordinateSystem(X+, Z+, Y-)"
    assert repr(CoordinateSystem("X+", "Y+", "Z+")) == "CoordinateSystem(X+, Y+, Z+)"
