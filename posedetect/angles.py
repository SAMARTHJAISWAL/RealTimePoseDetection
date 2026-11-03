"""Joint angle geometry. Pure math: no OpenCV or MediaPipe imports."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = Sequence[float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Directional angle at ``b`` from ray b→a to ray b→c, in [0, 360).

    Only the x and y coordinates are used. The result depends on the order of
    ``a`` and ``c``, so it distinguishes a joint bent one way from the other.
    """
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return angle % 360.0


def joint_angle(a: Point, b: Point, c: Point) -> float:
    """Interior angle at ``b`` between segments b–a and b–c, in [0, 180].

    180 means the three points are in a straight line (a fully extended limb).
    The result is the same for a pose and its mirror image.
    """
    angle = calculate_angle(a, b, c)
    return 360.0 - angle if angle > 180.0 else angle
