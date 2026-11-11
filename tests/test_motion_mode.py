import numpy as np
import pytest

from src.ego_motion import EgoMotionResult
from src.motion_mode import CameraMode, MotionModeDecider


def make_result(tx: float = 0.0, ty: float = 0.0, rot: float = 0.0, scale: float = 1.0, compensated: bool = True) -> EgoMotionResult:
    return EgoMotionResult(
        M=np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32),
        tx=tx,
        ty=ty,
        rotation_deg=rot,
        scale=scale,
        inlier_ratio=0.8,
        compensated=compensated,
    )


def test_motion_mode_static_remains_static():
    decider = MotionModeDecider(window_size=5, dynamic_threshold=2.5, static_threshold=1.0)
    # On first frame, is_transitioning must be False
    mode = decider.decide(make_result(tx=0.2, ty=0.3))
    assert mode == CameraMode.STATIC
    assert decider.is_transitioning is False

    for _ in range(9):
        mode = decider.decide(make_result(tx=0.2, ty=0.3))
        assert mode == CameraMode.STATIC
        assert decider.is_transitioning is False


def test_motion_mode_dynamic_transition():
    decider = MotionModeDecider(window_size=5, dynamic_threshold=2.5, static_threshold=1.0)
    # Warm up with 5 static frames
    for _ in range(5):
        decider.decide(make_result(tx=0.0, ty=0.0))
    assert decider.current_mode == CameraMode.STATIC

    # Frames 6, 7, 8: average climbs (0.72, 1.44, 2.16 < 2.5)
    for _ in range(3):
        mode = decider.decide(make_result(tx=3.0, ty=2.0))
        assert mode == CameraMode.STATIC

    # Frame 9: average reaches 2.88 >= 2.5 -> transitions!
    mode = decider.decide(make_result(tx=3.0, ty=2.0))
    assert mode == CameraMode.DYNAMIC
    assert decider.is_transitioning is True


def test_motion_mode_hysteresis():
    decider = MotionModeDecider(window_size=5, dynamic_threshold=2.5, static_threshold=1.0)

    # 1. Force into DYNAMIC mode
    for _ in range(6):
        decider.decide(make_result(tx=4.0, ty=0.0))
    assert decider.current_mode == CameraMode.DYNAMIC

    # 2. Feed intermediate motion (tx=1.8, ty=0.0 -> mag=1.8, between 1.0 and 2.5)
    for _ in range(10):
        mode = decider.decide(make_result(tx=1.8, ty=0.0))
        # Hysteresis should keep camera in DYNAMIC mode
        assert mode == CameraMode.DYNAMIC

    # 3. Now drop motion below 1.0 (tx=0.2, ty=0.1)
    for _ in range(5):
        mode = decider.decide(make_result(tx=0.2, ty=0.1))
    assert mode == CameraMode.STATIC


def test_motion_mode_overrides():
    # Forced static
    static_decider = MotionModeDecider(camera_mode="static")
    mode = static_decider.decide(make_result(tx=50.0, ty=50.0))
    assert mode == CameraMode.STATIC

    # Forced dynamic
    dynamic_decider = MotionModeDecider(camera_mode="dynamic")
    mode = dynamic_decider.decide(make_result(tx=0.0, ty=0.0))
    assert mode == CameraMode.DYNAMIC


def test_motion_mode_drift_scenario():
    """Validates camera behavior under slow mounting sag / wind drift vs accelerating drift."""
    decider = MotionModeDecider(window_size=6, dynamic_threshold=2.5, static_threshold=1.0)

    # 1. Slow drift (0.7 px/frame): benign mechanical creep/vibration
    # Must remain in STATIC mode without false alarms
    for _ in range(12):
        mode = decider.decide(make_result(tx=0.5, ty=0.5))  # magnitude ~ 0.71 px
        assert mode == CameraMode.STATIC

    # 2. Accelerating drift (2.8 px/frame): pan motion begins
    for _ in range(5):
        mode = decider.decide(make_result(tx=2.2, ty=1.8))  # magnitude ~ 2.84 px
    # Now window average crosses dynamic_threshold (2.5 px)
    mode = decider.decide(make_result(tx=2.2, ty=1.8))
    assert mode == CameraMode.DYNAMIC

    # 3. Intermediate drift (1.4 px/frame): decelerating pan
    # Hysteresis must hold camera in DYNAMIC mode (between 1.0 and 2.5)
    for _ in range(10):
        mode = decider.decide(make_result(tx=1.0, ty=1.0))  # magnitude ~ 1.41 px
        assert mode == CameraMode.DYNAMIC

    # 4. Stationary settlement (0.14 px/frame)
    # Once motion remains below 1.0 px across the window, reverts to STATIC
    for _ in range(6):
        decider.decide(make_result(tx=0.1, ty=0.1))
    assert decider.current_mode == CameraMode.STATIC


def test_motion_mode_zoom_only_triggers_dynamic():
    """Pure zoom on a PTZ camera (tx=0, ty=0) must trigger DYNAMIC mode."""
    decider = MotionModeDecider(window_size=5, dynamic_threshold=2.5, static_threshold=1.0)

    # 5 frames of stationary feed
    for _ in range(5):
        mode = decider.decide(make_result(tx=0.0, ty=0.0, scale=1.0))
        assert mode == CameraMode.STATIC

    # Now pure zoom begins (3.5% zoom: |1.035 - 1.0| * 100 = 3.5 > 2.5)
    for _ in range(4):
        decider.decide(make_result(tx=0.0, ty=0.0, scale=1.035))
    mode = decider.decide(make_result(tx=0.0, ty=0.0, scale=1.035))
    assert mode == CameraMode.DYNAMIC


def test_motion_mode_pan_latency():
    """Measures and asserts the exact transition latency for pan-only motion from an established static baseline."""
    window_size = 5
    dynamic_threshold = 2.5
    decider = MotionModeDecider(window_size=window_size, dynamic_threshold=dynamic_threshold, static_threshold=1.0)

    # Establish STATIC baseline (fill sliding window with zero motion)
    for _ in range(window_size):
        assert decider.decide(make_result(tx=0.0, ty=0.0)) == CameraMode.STATIC

    # Begin continuous pan of 3.5 px/frame
    pan_motion = 3.5
    transition_frame = None

    for frame_idx in range(1, 10):
        mode = decider.decide(make_result(tx=pan_motion, ty=0.0))
        if mode == CameraMode.DYNAMIC and transition_frame is None:
            transition_frame = frame_idx
            break

    # With window_size=5, avg_mag = (frame_idx * 3.5) / 5.
    # At frame 3: (3 * 3.5) / 5 = 2.1 < 2.5 (STATIC)
    # At frame 4: (4 * 3.5) / 5 = 2.8 >= 2.5 (DYNAMIC) -> latency is exactly frame 4!
    assert transition_frame == 4, f"Expected pan transition at frame 4, but occurred at {transition_frame}"


def test_motion_mode_zoom_latency():
    """Measures and asserts the exact transition latency for zoom-only motion from an established static baseline."""
    window_size = 5
    dynamic_threshold = 2.5
    decider = MotionModeDecider(window_size=window_size, dynamic_threshold=dynamic_threshold, static_threshold=1.0)

    # Establish STATIC baseline
    for _ in range(window_size):
        assert decider.decide(make_result(scale=1.0)) == CameraMode.STATIC

    # Begin continuous 3.5% zoom per frame (penalty: |1.035 - 1.0| * 100 = 3.5 px/frame)
    transition_frame = None
    for frame_idx in range(1, 10):
        mode = decider.decide(make_result(scale=1.035))
        if mode == CameraMode.DYNAMIC and transition_frame is None:
            transition_frame = frame_idx
            break

    # Latency for zoom metric must match pan metric (exactly 4 frames)
    assert transition_frame == 4, f"Expected zoom transition at frame 4, but occurred at {transition_frame}"

