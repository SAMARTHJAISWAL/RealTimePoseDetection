import pytest

from posedetect.smoothing import LabelSmoother


def test_majority_wins_over_single_glitch():
    s = LabelSmoother(window=5)
    for _ in range(4):
        s.update("T Pose")
    assert s.update("Unknown") == "T Pose"


def test_switches_once_new_label_dominates():
    s = LabelSmoother(window=4)
    for _ in range(4):
        s.update("a")
    results = [s.update("b") for _ in range(3)]
    assert results[-1] == "b"


def test_tie_prefers_most_recent():
    s = LabelSmoother(window=2)
    s.update("a")
    assert s.update("b") == "b"


def test_window_of_one_disables_smoothing():
    s = LabelSmoother(window=1)
    assert [s.update(x) for x in "abc"] == ["a", "b", "c"]


def test_invalid_window():
    with pytest.raises(ValueError):
        LabelSmoother(window=0)
