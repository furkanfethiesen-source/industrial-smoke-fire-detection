"""Unit tests for TemporalFilter and False-Positive Reduction."""

import numpy as np
import pytest

from src.detector import Detection
from src.temporal_filter import TemporalFilter, TrackedDetection, compute_iou
from collections import deque


def test_compute_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)  # Area 100
    box2 = (5.0, 0.0, 15.0, 10.0)  # Area 100, Inter: 5*10=50, Union: 150 -> IoU = 1/3
    assert abs(compute_iou(box1, box2) - (50.0 / 150.0)) < 1e-4

    box_disjoint = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box_disjoint) == 0.0


def test_temporal_filter_suppresses_transient_noise():
    """Single-frame transient false alarms (e.g. flash) must not be confirmed."""
    filter_engine = TemporalFilter(min_hits=3, optical_flow_enabled=False)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Transient reflection
    raw_det = [
        Detection(box=(50.0, 50.0, 100.0, 100.0), confidence=0.85, class_id=0, class_name="smoke", area=2500.0)
    ]
    verified = filter_engine.update(dummy_frame, raw_det)
    assert len(verified) == 0, "Single-frame detection must be suppressed (hits < min_hits)"


def test_temporal_filter_confirms_persistent_hazard():
    """Detections persisting for >= min_hits frames must be verified and tracked."""
    filter_engine = TemporalFilter(min_hits=3, optical_flow_enabled=False)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    box = (50.0, 50.0, 100.0, 100.0)
    det = Detection(box=box, confidence=0.85, class_id=1, class_name="fire", area=2500.0)

    # Frame 1 & 2: Not yet confirmed
    assert len(filter_engine.update(dummy_frame, [det])) == 0
    assert len(filter_engine.update(dummy_frame, [det])) == 0

    # Frame 3: Confirmed!
    verified = filter_engine.update(dummy_frame, [det])
    assert len(verified) == 1
    assert verified[0].class_name == "fire"
    assert "track_id" in verified[0].extra
    assert verified[0].extra["hits"] == 3


def test_temporal_filter_roi_mode_computes_per_roi(monkeypatch):
    """ROI mode must compute optical flow ONLY within bounding boxes, not on full frame."""
    import cv2

    filter_engine = TemporalFilter(
        min_hits=1,
        optical_flow_enabled=True,
        optical_flow_mode="roi",
    )
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box coordinates: (50, 50, 150, 150) -> width=100, height=100
    box = (50.0, 50.0, 150.0, 150.0)
    # Generate textured pattern inside ROI that shifts upward
    np.random.seed(42)
    texture = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.0)

    frame1[50:150, 50:150, 0] = texture
    frame1[50:150, 50:150, 1] = texture
    frame1[50:150, 50:150, 2] = texture

    # Shift upward by 4 pixels in frame 2
    frame2[46:146, 50:150, 0] = texture
    frame2[46:146, 50:150, 1] = texture
    frame2[46:146, 50:150, 2] = texture

    det = Detection(box=box, confidence=0.9, class_id=0, class_name="smoke", area=10000.0)

    # Frame 1: Initialize previous frame
    filter_engine.update(frame1, [det])
    assert filter_engine.current_flow is None, "ROI mode must NOT compute full frame flow in update"

    recorded_shapes = []
    original_farneback = cv2.calcOpticalFlowFarneback

    def spy_farneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags):
        recorded_shapes.append((prev.shape, next.shape))
        return original_farneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", spy_farneback)

    # Frame 2: Process detection with motion
    verified = filter_engine.update(frame2, [det])

    assert filter_engine.current_flow is None, "ROI mode current_flow must remain None"
    assert len(recorded_shapes) == 1, "Optical flow should have been computed exactly once for the single ROI"
    roi_prev_shape, roi_next_shape = recorded_shapes[0]
    assert roi_prev_shape == (100, 100), f"Expected ROI shape (100, 100), got {roi_prev_shape}"
    assert roi_next_shape == (100, 100), f"Expected ROI shape (100, 100), got {roi_next_shape}"

    # Verify that _evaluate_motion works with per-ROI flow directly
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    filter_engine.prev_gray_frame = gray1
    motion_score = filter_engine._evaluate_motion(box, gray2)
    assert isinstance(motion_score, float)
    assert motion_score > 0.5, "Upward motion within ROI must yield positive motion score"


def test_temporal_filter_distinguishes_upward_and_downward_motion():
    """Downward motion (e.g. falling debris or water) must be rejected or have low score compared to rising smoke."""
    import cv2
    filter_engine = TemporalFilter(min_hits=1, optical_flow_enabled=True, optical_flow_mode="roi")
    box = (50.0, 50.0, 150.0, 150.0)

    np.random.seed(42)
    texture = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.0)

    # Frame 1: Base texture
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[50:150, 50:150, 0] = texture
    frame1[50:150, 50:150, 1] = texture
    frame1[50:150, 50:150, 2] = texture

    # Frame Down: Texture shifted DOWNWARD by 4 pixels (y increases downward in image coords)
    frame_down = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_down[54:154, 50:150, 0] = texture
    frame_down[54:154, 50:150, 1] = texture
    frame_down[54:154, 50:150, 2] = texture

    det = Detection(box=box, confidence=0.9, class_id=0, class_name="smoke", area=10000.0)
    filter_engine.update(frame1, [det])

    # Downward motion should produce a motion score low enough to be rejected or significantly lower
    gray_down = cv2.cvtColor(frame_down, cv2.COLOR_BGR2GRAY)
    down_score = filter_engine._evaluate_motion(box, gray_down)
    # The downward motion score should NOT pass the acceptance threshold
    assert down_score < 0.35, f"Downward motion score {down_score} should be < 0.35"


def test_temporal_filter_centroid_distance_association():
    """Deforming smoke whose bounding box expands or shifts (low IoU) should still associate by centroid."""
    filter_engine = TemporalFilter(min_hits=3, optical_flow_enabled=False, iou_match_threshold=0.25)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box 1: (50, 50, 100, 100) -> center (75, 75), area 2500
    det1 = Detection(box=(50.0, 50.0, 100.0, 100.0), confidence=0.85, class_id=0, class_name="smoke", area=2500.0)
    filter_engine.update(dummy_frame, [det1])

    # Box 2: Rapidly expands to (30, 30, 140, 140) -> center (85, 85).
    # IoU is (50*50) / (110*110) = 2500 / 12100 = ~0.206 (< 0.25 IoU threshold!)
    # But centroid distance is only ~14 pixels. Centroid matching must maintain track_id=1.
    det2 = Detection(box=(30.0, 30.0, 140.0, 140.0), confidence=0.85, class_id=0, class_name="smoke", area=12100.0)
    filter_engine.update(dummy_frame, [det2])

    assert 1 in filter_engine.tracks
    assert filter_engine.tracks[1].hits == 2, "Track should have matched via centroid distance despite IoU < 0.25"



def test_temporal_filter_full_mode_backwards_compatibility():
    """Full mode must compute optical flow on the entire frame as original behavior."""
    filter_engine = TemporalFilter(
        min_hits=1,
        optical_flow_enabled=True,
        optical_flow_mode="full",
    )
    frame1 = np.zeros((100, 120, 3), dtype=np.uint8)
    frame2 = np.ones((100, 120, 3), dtype=np.uint8) * 128

    filter_engine.update(frame1, [])
    assert filter_engine.current_flow is None

    filter_engine.update(frame2, [])
    assert filter_engine.current_flow is not None
    assert filter_engine.current_flow.shape == (100, 120, 2)


def test_history_boxes_does_not_grow_unbounded():
    """history_boxes deque'si queue_size ile sınırlı olmalı — bellek sızıntısı olmamalı.

    queue_size=5 ile 20 kare işlense bile her track'in history_boxes boyutu 5'i geçmemeli.
    """
    queue_size = 5
    filter_engine = TemporalFilter(
        queue_size=queue_size,
        min_hits=1,
        optical_flow_enabled=False,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(box=(50.0, 50.0, 100.0, 100.0), confidence=0.9, class_id=0, class_name="smoke", area=2500.0)

    for _ in range(20):
        filter_engine.update(dummy_frame, [det])

    assert len(filter_engine.tracks) == 1
    track = list(filter_engine.tracks.values())[0]
    assert len(track.history_boxes) <= queue_size, (
        f"history_boxes boyutu {len(track.history_boxes)} > queue_size={queue_size}: bellek sızıntısı!"
    )


def test_expansion_trend_positive_on_growing_box():
    """Büyüyen kutu için expansion_trend pozitif olmalı ve extra'ya yazılmalı."""
    filter_engine = TemporalFilter(
        min_hits=2,
        optical_flow_enabled=False,
        queue_size=10,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Küçük kutu — center (100, 100), 100×100
    det_small = Detection(box=(50.0, 50.0, 150.0, 150.0), confidence=0.9, class_id=1, class_name="fire", area=10000.0)
    filter_engine.update(dummy_frame, [det_small])

    # Frame 2: Büyük kutu — center (100, 100), 200×200 (aynı merkez → centroid dist=0, IoU > 0.25)
    det_large = Detection(box=(0.0, 0.0, 200.0, 200.0), confidence=0.9, class_id=1, class_name="fire", area=40000.0)
    verified = filter_engine.update(dummy_frame, [det_large])

    assert len(verified) == 1, (
        f"2 hit ile onaylanmalı; got {len(verified)}. "
        f"Tracks: {[(t, v.hits, v.last_box) for t, v in filter_engine.tracks.items()]}"
    )
    track = list(filter_engine.tracks.values())[0]

    # expansion_trend = (40000 - 10000) / 10000 = 3.0 (300% büyüme)
    assert track.expansion_trend > 0, f"Büyüyen kutu için expansion_trend pozitif olmalı: {track.expansion_trend}"
    assert "expansion_trend" in verified[0].extra, "expansion_trend extra sözlüğünde olmalı"
    assert verified[0].extra["expansion_trend"] > 0


def test_temporal_filter_static_mode_ignores_M_as_baseline():
    """STATIC mode must ignore M and never warp track boxes, preserving stationary baseline."""
    filter_engine = TemporalFilter(min_hits=1, optical_flow_enabled=False)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = Detection(box=(100.0, 100.0, 150.0, 150.0), confidence=0.9, class_id=1, class_name="fire", area=2500.0)
    filter_engine.update(dummy_frame, [det], camera_mode="static")

    # Frame 2: Large M passed with camera_mode="static"
    M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32)
    filter_engine.update(dummy_frame, [], M=M, camera_mode="static")

    track = list(filter_engine.tracks.values())[0]
    # Box should NOT be warped because mode is static
    assert track.last_box == (100.0, 100.0, 150.0, 150.0)


def test_temporal_filter_dynamic_mode_applies_M():
    """DYNAMIC mode must apply M to warp existing tracks so camera motion doesn't drop tracks."""
    filter_engine = TemporalFilter(min_hits=1, optical_flow_enabled=False)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = Detection(box=(100.0, 100.0, 150.0, 150.0), confidence=0.9, class_id=1, class_name="fire", area=2500.0)
    filter_engine.update(dummy_frame, [det], camera_mode="dynamic")

    # Frame 2: M passed with camera_mode="dynamic"
    M = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    filter_engine.update(dummy_frame, [], M=M, camera_mode="dynamic")

    track = list(filter_engine.tracks.values())[0]
    # Box should be warped: (100+30, 100+20, 150+30, 150+20) = (130, 120, 180, 170)
    assert pytest.approx(track.last_box[0], abs=0.5) == 130.0
    assert pytest.approx(track.last_box[1], abs=0.5) == 120.0
    assert pytest.approx(track.last_box[2], abs=0.5) == 180.0
    assert pytest.approx(track.last_box[3], abs=0.5) == 170.0


