"""Rule-based pose classification from joint angles.

Every rule uses interior angles (0–180°), so a pose is recognised the same way
whichever side of the body leads and whether or not the frame is mirrored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum

from posedetect.angles import joint_angle
from posedetect.landmarks import Landmark as L

# Landmarks every rule depends on. If any is poorly visible, the angles are
# guesses and the frame is reported as Unknown rather than misclassified.
REQUIRED_LANDMARKS = (
    L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_ELBOW, L.RIGHT_ELBOW,
    L.LEFT_WRIST, L.RIGHT_WRIST, L.LEFT_HIP, L.RIGHT_HIP,
    L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE,
)

STRAIGHT_ARM = 165.0
STRAIGHT_LEG = 165.0
STRAIGHT_LEG_T_POSE = 160.0
# Warrior II leans the torso slightly, which opens the shoulder angle past 110°.
SHOULDER_HORIZONTAL = (75.0, 115.0)
WARRIOR_FRONT_KNEE = (90.0, 120.0)
TREE_BENT_KNEE = (25.0, 45.0)


class Pose(str, Enum):
    WARRIOR_II = "Warrior II Pose"
    T_POSE = "T Pose"
    TREE = "Tree Pose"
    UNKNOWN = "Unknown Pose"


@dataclass(frozen=True)
class JointAngles:
    left_elbow: float
    right_elbow: float
    left_shoulder: float
    right_shoulder: float
    left_knee: float
    right_knee: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_angles(landmarks: Sequence[Sequence[float]]) -> JointAngles:
    """Compute the six joint angles used for classification.

    ``landmarks`` is the 33-point BlazePose list; each point is ``(x, y, ...)``
    in any consistent unit (pixels or normalised).
    """
    p = landmarks
    return JointAngles(
        left_elbow=joint_angle(p[L.LEFT_SHOULDER], p[L.LEFT_ELBOW], p[L.LEFT_WRIST]),
        right_elbow=joint_angle(p[L.RIGHT_SHOULDER], p[L.RIGHT_ELBOW], p[L.RIGHT_WRIST]),
        left_shoulder=joint_angle(p[L.LEFT_ELBOW], p[L.LEFT_SHOULDER], p[L.LEFT_HIP]),
        right_shoulder=joint_angle(p[L.RIGHT_ELBOW], p[L.RIGHT_SHOULDER], p[L.RIGHT_HIP]),
        left_knee=joint_angle(p[L.LEFT_HIP], p[L.LEFT_KNEE], p[L.LEFT_ANKLE]),
        right_knee=joint_angle(p[L.RIGHT_HIP], p[L.RIGHT_KNEE], p[L.RIGHT_ANKLE]),
    )


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _one_straight_one_bent(a: JointAngles, bent: tuple[float, float]) -> bool:
    left_straight = a.left_knee >= STRAIGHT_LEG
    right_straight = a.right_knee >= STRAIGHT_LEG
    return (left_straight and _within(a.right_knee, bent)) or (
        right_straight and _within(a.left_knee, bent)
    )


def _arms_out_horizontal(a: JointAngles) -> bool:
    return (
        a.left_elbow >= STRAIGHT_ARM
        and a.right_elbow >= STRAIGHT_ARM
        and _within(a.left_shoulder, SHOULDER_HORIZONTAL)
        and _within(a.right_shoulder, SHOULDER_HORIZONTAL)
    )


def is_warrior_ii(a: JointAngles) -> bool:
    return _arms_out_horizontal(a) and _one_straight_one_bent(a, WARRIOR_FRONT_KNEE)


def is_t_pose(a: JointAngles) -> bool:
    return (
        _arms_out_horizontal(a)
        and a.left_knee >= STRAIGHT_LEG_T_POSE
        and a.right_knee >= STRAIGHT_LEG_T_POSE
    )


def is_tree(a: JointAngles) -> bool:
    return _one_straight_one_bent(a, TREE_BENT_KNEE)


RULES = (
    (Pose.WARRIOR_II, is_warrior_ii),
    (Pose.T_POSE, is_t_pose),
    (Pose.TREE, is_tree),
)


def classify(angles: JointAngles) -> Pose:
    """Return the first pose whose rule matches, or ``Pose.UNKNOWN``."""
    for pose, rule in RULES:
        if rule(angles):
            return pose
    return Pose.UNKNOWN


def landmarks_visible(visibility: Sequence[float], threshold: float = 0.5) -> bool:
    """True if every landmark the rules need has visibility above ``threshold``."""
    return all(visibility[i] >= threshold for i in REQUIRED_LANDMARKS)
