"""OpenCV overlays: skeleton, pose label, joint angles, FPS."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from posedetect.classifier import JointAngles, Pose
from posedetect.landmarks import POSE_CONNECTIONS, Landmark as L

GREEN = (80, 220, 100)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
DARK = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX

ANGLE_JOINTS = {
    "left_elbow": L.LEFT_ELBOW,
    "right_elbow": L.RIGHT_ELBOW,
    "left_shoulder": L.LEFT_SHOULDER,
    "right_shoulder": L.RIGHT_SHOULDER,
    "left_knee": L.LEFT_KNEE,
    "right_knee": L.RIGHT_KNEE,
}


def _px(point: Sequence[float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def draw_skeleton(
    frame: np.ndarray,
    points: Sequence[Sequence[float]],
    visibility: Sequence[float],
    min_visibility: float = 0.5,
) -> None:
    scale = max(1, frame.shape[0] // 480)
    for a, b in POSE_CONNECTIONS:
        if visibility[a] >= min_visibility and visibility[b] >= min_visibility:
            cv2.line(frame, _px(points[a]), _px(points[b]), WHITE, 2 * scale, cv2.LINE_AA)
    for point, vis in zip(points, visibility):
        if vis >= min_visibility:
            cv2.circle(frame, _px(point), 3 * scale, RED, -1, cv2.LINE_AA)


def _text_with_background(
    frame: np.ndarray, text: str, origin: tuple[int, int], color, scale: float, thickness: int
) -> None:
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    pad = max(4, int(6 * scale))
    cv2.rectangle(frame, (x - pad, y - h - pad), (x + w + pad, y + baseline + pad), DARK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_label(frame: np.ndarray, pose: Pose | None) -> None:
    scale = frame.shape[0] / 640
    if pose is None:
        text, color = "No person detected", YELLOW
    else:
        text, color = pose.value, RED if pose is Pose.UNKNOWN else GREEN
    _text_with_background(frame, text, (int(16 * scale), int(44 * scale)), color, 1.1 * scale, max(1, int(2 * scale)))


def draw_angles(frame: np.ndarray, points: Sequence[Sequence[float]], angles: JointAngles) -> None:
    scale = frame.shape[0] / 640
    for name, value in angles.as_dict().items():
        x, y = _px(points[ANGLE_JOINTS[name]])
        _text_with_background(
            frame, f"{value:.0f}", (x + int(8 * scale), y), YELLOW, 0.5 * scale, max(1, int(scale))
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    scale = frame.shape[0] / 640
    text = f"{fps:4.1f} FPS"
    (w, _), _ = cv2.getTextSize(text, FONT, 0.6 * scale, 1)
    _text_with_background(
        frame, text, (frame.shape[1] - w - int(16 * scale), int(32 * scale)), WHITE, 0.6 * scale, max(1, int(scale))
    )
