"""Command-line entry point: ``python -m posedetect``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from posedetect.classifier import Pose, classify, compute_angles, landmarks_visible
from posedetect.detector import MODELS, PoseDetector
from posedetect.drawing import draw_angles, draw_fps, draw_label, draw_skeleton
from posedetect.smoothing import LabelSmoother

WINDOW = "Pose Classification"
QUIT_KEYS = (27, ord("q"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="posedetect",
        description="Detect body landmarks and classify yoga/fitness poses in real time.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    source.add_argument("--video", type=Path, help="classify a video file instead of the webcam")
    source.add_argument("--image", type=Path, help="classify a single image")
    parser.add_argument("--model", choices=MODELS, default="full",
                        help="pose model: lite is fastest, heavy is most accurate (default: full)")
    parser.add_argument("--model-path", type=Path, help="use a local .task model file instead of downloading")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="minimum detection/tracking confidence (default: 0.5)")
    parser.add_argument("--show-angles", action="store_true", help="draw joint angles on the skeleton")
    parser.add_argument("--no-mirror", action="store_true", help="don't mirror the webcam image")
    parser.add_argument("--smoothing", type=int, default=8,
                        help="frames to vote over for a steady label; 1 disables (default: 8)")
    parser.add_argument("--output", type=Path, help="save the annotated image or video to this path")
    parser.add_argument("--no-display", action="store_true", help="don't open a window (for scripts and servers)")
    return parser.parse_args(argv)


def annotate(frame: np.ndarray, detector: PoseDetector, show_angles: bool) -> tuple[Pose | None, np.ndarray]:
    """Detect and classify one frame; draws on ``frame`` in place and returns the raw pose."""
    detection = detector.detect(frame)
    if detection is None:
        return None, frame
    draw_skeleton(frame, detection.points, detection.visibility)
    if not landmarks_visible(detection.visibility):
        return Pose.UNKNOWN, frame
    angles = compute_angles(detection.points)
    if show_angles:
        draw_angles(frame, detection.points, angles)
    return classify(angles), frame


def resize_to_height(frame: np.ndarray, height: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    return cv2.resize(frame, (int(w * height / h), height))


def run_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Error: could not read image {args.image}", file=sys.stderr)
        return 1
    with PoseDetector(args.model, args.model_path, video=False, min_confidence=args.min_confidence) as detector:
        pose, frame = annotate(frame, detector, args.show_angles)
    draw_label(frame, pose)
    print(f"Pose: {pose.value if pose else 'no person detected'}")
    if args.output:
        cv2.imwrite(str(args.output), frame)
        print(f"Saved {args.output}")
    if not args.no_display:
        cv2.imshow(WINDOW, resize_to_height(frame))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(args: argparse.Namespace) -> int:
    is_camera = args.video is None
    capture = cv2.VideoCapture(args.camera if is_camera else str(args.video))
    if not capture.isOpened():
        what = f"camera {args.camera}" if is_camera else f"video {args.video}"
        print(f"Error: could not open {what}.", file=sys.stderr)
        if is_camera and sys.platform == "darwin":
            print("On macOS, allow camera access for your terminal in "
                  "System Settings → Privacy & Security → Camera.", file=sys.stderr)
        return 1
    if is_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    mirror = is_camera and not args.no_mirror
    show_angles = args.show_angles
    smoother: LabelSmoother[Pose | None] = LabelSmoother(max(1, args.smoothing))
    writer = None
    fps = 0.0
    last = time.perf_counter()

    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("Press q or Esc to quit, a to toggle joint angles.")

    try:
        with PoseDetector(args.model, args.model_path, video=True, min_confidence=args.min_confidence) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if is_camera:
                        print("Error: cannot read from camera.", file=sys.stderr)
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                frame = resize_to_height(frame)

                pose, frame = annotate(frame, detector, show_angles)
                draw_label(frame, smoother.update(pose))

                now = time.perf_counter()
                instant = 1.0 / max(now - last, 1e-6)
                fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
                last = now
                draw_fps(frame, fps)

                if args.output:
                    if writer is None:
                        h, w = frame.shape[:2]
                        rate = capture.get(cv2.CAP_PROP_FPS) or 30
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
                    writer.write(frame)

                if not args.no_display:
                    cv2.imshow(WINDOW, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in QUIT_KEYS:
                        break
                    if key == ord("a"):
                        show_angles = not show_angles
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Saved {args.output}")
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.image:
        return run_image(args)
    return run_stream(args)


if __name__ == "__main__":
    sys.exit(main())
