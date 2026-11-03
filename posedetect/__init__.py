"""Real-time human pose detection and yoga/fitness pose classification."""

from posedetect.angles import calculate_angle, joint_angle
from posedetect.classifier import JointAngles, Pose, classify, compute_angles

__all__ = [
    "JointAngles",
    "Pose",
    "calculate_angle",
    "classify",
    "compute_angles",
    "joint_angle",
]

__version__ = "2.0.0"
