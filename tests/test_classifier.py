import math

import pytest

from posedetect.classifier import (
    JointAngles,
    Pose,
    classify,
    compute_angles,
    landmarks_visible,
)
from posedetect.landmarks import Landmark as L


def angles(**overrides) -> JointAngles:
    """A T pose by default; override individual joints."""
    base = dict(
        left_elbow=178, right_elbow=176,
        left_shoulder=92, right_shoulder=95,
        left_knee=175, right_knee=172,
    )
    base.update(overrides)
    return JointAngles(**base)


def test_t_pose():
    assert classify(angles()) is Pose.T_POSE


@pytest.mark.parametrize("bent", ["left_knee", "right_knee"])
def test_warrior_ii_with_either_leg_forward(bent):
    assert classify(angles(**{bent: 105})) is Pose.WARRIOR_II


@pytest.mark.parametrize("bent", ["left_knee", "right_knee"])
def test_tree_with_either_leg_raised(bent):
    a = angles(left_elbow=60, right_elbow=60, left_shoulder=160, right_shoulder=160, **{bent: 35})
    assert classify(a) is Pose.TREE


def test_tree_does_not_depend_on_arms():
    assert classify(angles(**{"right_knee": 30})) is Pose.TREE


def test_arms_down_standing_is_unknown():
    assert classify(angles(left_shoulder=15, right_shoulder=12)) is Pose.UNKNOWN


def test_warrior_ii_with_leaning_torso():
    # Measured from assets/warrior_ii.jpg with the full model.
    a = angles(left_elbow=179, right_elbow=177, left_shoulder=85, right_shoulder=112,
               left_knee=176, right_knee=105)
    assert classify(a) is Pose.WARRIOR_II


def test_arms_raised_in_a_v_is_unknown():
    # Measured from assets/unknown_arms_raised.jpg.
    assert classify(angles(left_shoulder=128, right_shoulder=123)) is Pose.UNKNOWN


def test_arms_raised_overhead_is_unknown():
    assert classify(angles(left_shoulder=150, right_shoulder=150)) is Pose.UNKNOWN


def test_bent_elbows_break_t_pose():
    assert classify(angles(left_elbow=120)) is Pose.UNKNOWN


def test_squat_is_unknown():
    assert classify(angles(left_knee=100, right_knee=100)) is Pose.UNKNOWN


def _point_at(origin, length, degrees):
    """Point ``length`` away from ``origin``; 0° is +x, 90° is +y (down in image space)."""
    r = math.radians(degrees)
    return (origin[0] + length * math.cos(r), origin[1] + length * math.sin(r))


def make_skeleton(arm_dir_left=180, arm_dir_right=0, left_shin=90, right_shin=90):
    """Build a 33-point skeleton in image coordinates (y grows downward)."""
    pts = [(0.0, 0.0)] * 33
    pts[L.LEFT_SHOULDER], pts[L.RIGHT_SHOULDER] = (-40, 0), (40, 0)
    pts[L.LEFT_HIP], pts[L.RIGHT_HIP] = (-30, 200), (30, 200)
    pts[L.LEFT_ELBOW] = _point_at(pts[L.LEFT_SHOULDER], 80, arm_dir_left)
    pts[L.LEFT_WRIST] = _point_at(pts[L.LEFT_ELBOW], 80, arm_dir_left)
    pts[L.RIGHT_ELBOW] = _point_at(pts[L.RIGHT_SHOULDER], 80, arm_dir_right)
    pts[L.RIGHT_WRIST] = _point_at(pts[L.RIGHT_ELBOW], 80, arm_dir_right)
    pts[L.LEFT_KNEE], pts[L.RIGHT_KNEE] = (-30, 300), (30, 300)
    pts[L.LEFT_ANKLE] = _point_at(pts[L.LEFT_KNEE], 100, left_shin)
    pts[L.RIGHT_ANKLE] = _point_at(pts[L.RIGHT_KNEE], 100, right_shin)
    return pts


def test_compute_angles_on_t_pose_skeleton():
    a = compute_angles(make_skeleton())
    assert a.left_elbow == pytest.approx(180)
    assert a.right_elbow == pytest.approx(180)
    assert 80 <= a.left_shoulder <= 110
    assert 80 <= a.right_shoulder <= 110
    assert a.left_knee == pytest.approx(180)
    assert classify(a) is Pose.T_POSE


def test_skeleton_and_its_mirror_classify_the_same():
    # Right shin folded back up toward the hip, as in tree pose.
    skeleton = make_skeleton(arm_dir_left=90, arm_dir_right=90, right_shin=-60)
    mirrored = [(-x, y) for x, y in skeleton]
    mirrored[L.LEFT_SHOULDER::2], mirrored[L.RIGHT_SHOULDER::2] = (
        mirrored[L.RIGHT_SHOULDER::2], mirrored[L.LEFT_SHOULDER::2]
    )
    assert classify(compute_angles(skeleton)) is Pose.TREE
    assert classify(compute_angles(mirrored)) is Pose.TREE


def test_landmarks_visible():
    vis = [1.0] * 33
    assert landmarks_visible(vis)
    vis[L.LEFT_ANKLE] = 0.2
    assert not landmarks_visible(vis)
    vis = [1.0] * 33
    vis[L.NOSE] = 0.0  # not used by any rule
    assert landmarks_visible(vis)
