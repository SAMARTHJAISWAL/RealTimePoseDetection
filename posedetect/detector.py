"""MediaPipe Pose Landmarker wrapper and model download."""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task"
)
MODELS = ("lite", "full", "heavy")


def model_cache_dir() -> Path:
    override = os.environ.get("POSEDETECT_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "posedetect"


def ensure_model(name: str) -> Path:
    """Return the local path of a Pose Landmarker model, downloading it once."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    target = model_cache_dir() / f"pose_landmarker_{name}.task"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(name=name)
    print(f"Downloading pose model '{name}' to {target} ...")
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


@dataclass(frozen=True)
class Detection:
    """One detected person. ``points`` are ``(x_px, y_px, z)`` per landmark."""

    points: list[tuple[float, float, float]]
    visibility: list[float]


class PoseDetector:
    """Detects a single person's 33 body landmarks in BGR frames.

    ``video=True`` enables MediaPipe's tracking between frames, which is faster
    and steadier for webcam or video input; use ``video=False`` for stills.
    """

    def __init__(
        self,
        model: str = "full",
        model_path: str | Path | None = None,
        video: bool = True,
        min_confidence: float = 0.5,
    ) -> None:
        path = Path(model_path) if model_path else ensure_model(model)
        self._video = video
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            # VIDEO mode requires strictly increasing timestamps.
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        return Detection(
            points=[(lm.x * width, lm.y * height, lm.z * width) for lm in lms],
            visibility=[lm.visibility if lm.visibility is not None else 1.0 for lm in lms],
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
