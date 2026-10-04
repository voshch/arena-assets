from arena_assets.utils.geom import BoundingBox


def test_accessors_and_volume():
    box = BoundingBox(((-1.0, 1.0), (0.0, 2.0), (3.0, 4.0)))
    assert (box.min_x, box.max_x) == (-1.0, 1.0)
    assert (box.min_y, box.max_y) == (0.0, 2.0)
    assert (box.min_z, box.max_z) == (3.0, 4.0)
    assert box.volume == 4.0


def test_empty():
    assert BoundingBox.empty().volume == 0.0


def test_round():
    box = BoundingBox(((0.123456, 1.0), (0.0, 1.0), (0.0, 1.0))).round(2)
    assert box.min_x == 0.12
