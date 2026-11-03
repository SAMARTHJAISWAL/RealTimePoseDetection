import pytest

from posedetect.angles import calculate_angle, joint_angle


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        ((1, 0), (0, 0), (0, 1), 90),
        ((0, 1), (0, 0), (1, 0), 270),
        ((-1, 0), (0, 0), (1, 0), 180),
        ((1, 0), (0, 0), (1, 1), 45),
    ],
)
def test_calculate_angle_is_directional(a, b, c, expected):
    assert calculate_angle(a, b, c) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        ((1, 0), (0, 0), (0, 1), 90),
        ((0, 1), (0, 0), (1, 0), 90),
        ((-1, 0), (0, 0), (1, 0), 180),
        ((1, 0), (0, 0), (1, 0), 0),
    ],
)
def test_joint_angle_is_interior_and_order_independent(a, b, c, expected):
    assert joint_angle(a, b, c) == pytest.approx(expected)
    assert joint_angle(c, b, a) == pytest.approx(expected)


def test_extra_coordinates_are_ignored():
    assert joint_angle((1, 0, 5.0), (0, 0, -3.0), (0, 1, 9.0)) == pytest.approx(90)
