"""Runs the real MediaPipe model on the sample images.

Catches breaking changes in MediaPipe itself, which the unit tests can't see.
Skip with ``pytest -m "not model"`` when offline.
"""

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from posedetect.__main__ import annotate, main  # noqa: E402
from posedetect.classifier import Pose  # noqa: E402
from posedetect.detector import PoseDetector  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def detector():
    with PoseDetector("full", video=False) as d:
        yield d


@pytest.mark.parametrize(
    "image, expected",
    [
        ("warrior_ii.jpg", Pose.WARRIOR_II),
        ("unknown_arms_raised.jpg", Pose.UNKNOWN),
        ("unknown_transition.jpg", Pose.UNKNOWN),
    ],
)
def test_sample_images(detector, image, expected):
    frame = cv2.imread(str(ASSETS / image))
    pose, _ = annotate(frame, detector, show_angles=True)
    assert pose is expected


def test_cli_image_mode(tmp_path, capsys):
    out = tmp_path / "annotated.jpg"
    code = main(["--image", str(ASSETS / "warrior_ii.jpg"), "--no-display", "--output", str(out)])
    assert code == 0
    assert "Pose: Warrior II Pose" in capsys.readouterr().out
    assert cv2.imread(str(out)) is not None


def test_video_mode_tracks_across_frames():
    frame = cv2.imread(str(ASSETS / "warrior_ii.jpg"))
    with PoseDetector("full", video=True) as d:
        poses = [annotate(frame.copy(), d, show_angles=False)[0] for _ in range(5)]
    assert poses == [Pose.WARRIOR_II] * 5
