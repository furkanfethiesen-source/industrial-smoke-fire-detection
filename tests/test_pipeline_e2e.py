import cv2
import numpy as np
import pytest

from src.detector import Detection, FrameDetections
from src.main import IndustrialPipeline
from src.risk_analyzer import RiskLevel


def create_textured_scene(width=640, height=480, seed=42) -> np.ndarray:
    """Generates a textured scene with corner features."""
    np.random.seed(seed)
    img = np.random.randint(60, 190, (height, width, 3), dtype=np.uint8)
    for i in range(25, width - 25, 35):
        for j in range(25, height - 25, 35):
            cv2.rectangle(img, (i, j), (i + 14, j + 14), (240, 240, 240), -1)
            cv2.circle(img, (i + 20, j + 20), 5, (20, 20, 20), -1)
    return img


def test_process_frame_integration():
    """Verify that IndustrialPipeline.process_frame integrates all components seamlessly."""
    config = {
        "detector": {"conf_threshold": 0.25, "img_size": 640},
        "temporal_filter": {"min_hits": 2, "optical_flow_enabled": False},
        "camera_motion": {"camera_mode": "auto"},
        "risk_analyzer": {"warning_growth_rate": 0.30},
        "alert_connector": {"enabled": False},
    }
    pipeline = IndustrialPipeline(config=config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated, assessment, raw_results = pipeline.process_frame(frame, timestamp=1.0)
    assert annotated.shape == (480, 640, 3)
    assert assessment.level == RiskLevel.NORMAL
    assert isinstance(raw_results, FrameDetections)


def test_e2e_pan_zoom_static_smoke_stays_monitoring(monkeypatch):
    """End-to-end synthetic sequence: camera panning and zooming while physical smoke is stationary.

    Validates:
    1. Ego-motion compensates pan and scale.
    2. MotionMode switches to DYNAMIC.
    3. TemporalFilter confirms smoke track without dropping.
    4. Cumulative scale normalization keeps dA/dt ≈ 0 (Risk stays MONITORING, no false CRITICAL/WARNING).
    5. Spread direction stays stable despite camera translation.
    """
    config = {
        "detector": {"conf_threshold": 0.25, "img_size": 640},
        "temporal_filter": {"min_hits": 3, "optical_flow_enabled": False},
        "camera_motion": {
            "camera_mode": "auto",
            "window_size": 5,
            "dynamic_threshold": 1.5,
            "static_threshold": 0.8,
        },
        "risk_analyzer": {
            "warning_growth_rate": 0.30,
            "critical_growth_rate": 1.50,
            "history_window_sec": 0.4,
            "cooldown_frames": 3,
        },
        "alert_connector": {"enabled": False},
    }
    pipeline = IndustrialPipeline(config=config)

    w, h = 640, 480
    bg = create_textured_scene(w, h)

    # Physical smoke starts centered at optical center (320, 240) with base size 70x70
    center_x, center_y = 320.0, 240.0
    base_size = 70.0
    pan_x, pan_y = 2.0, 1.0  # Camera pans right and down
    zoom_step = 1.015        # 1.5% optical zoom per frame

    # Generate 18 sequential frames
    num_frames = 18
    frames = []
    current_boxes = []

    for k in range(num_frames):
        # Compounding camera transformation relative to world
        s_k = zoom_step ** k
        tx_k = pan_x * k
        ty_k = pan_y * k

        # Camera frame transformation: zoom around optical center (center_x, center_y) + translation
        M_cam = cv2.getRotationMatrix2D((center_x, center_y), 0.0, s_k)
        M_cam[0, 2] += tx_k
        M_cam[1, 2] += ty_k

        # Warp scene
        warped_frame = cv2.warpAffine(bg, M_cam, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Smoke location on camera screen:
        # Optical center zoom scales around (center_x, center_y), then translates by (tx_k, ty_k)
        scr_cx = center_x + tx_k
        scr_cy = center_y + ty_k
        scr_size = base_size * s_k

        box = (
            float(scr_cx - scr_size / 2.0),
            float(scr_cy - scr_size / 2.0),
            float(scr_cx + scr_size / 2.0),
            float(scr_cy + scr_size / 2.0),
        )

        # Draw visual smoke patch onto frame
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(warped_frame, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), (180, 180, 180), -1)

        frames.append(warped_frame)
        current_boxes.append(box)

    # Mock detector to simulate reliable detection of the synthetic smoke box in each frame
    frame_idx_container = [0]

    def mock_detect(frame):
        idx = frame_idx_container[0]
        box = current_boxes[idx]
        area = (box[2] - box[0]) * (box[3] - box[1])
        det = Detection(
            box=box,
            confidence=0.92,
            class_id=0,
            class_name="smoke",
            area=area,
        )
        return FrameDetections(
            detections=[det],
            inference_time_ms=5.0,
            fps=30.0,
            frame_width=w,
            frame_height=h,
        )

    monkeypatch.setattr(pipeline.detector, "detect", mock_detect)

    # Record frame-by-frame telemetry for full sequence verification
    records = []
    for idx, frame in enumerate(frames):
        frame_idx_container[0] = idx
        timestamp = 1.0 + idx * 0.1
        annotated, assessment, _ = pipeline.process_frame(frame, timestamp=timestamp)
        records.append({
            "frame": idx,
            "timestamp": timestamp,
            "mode": pipeline.motion_mode.current_mode.value,
            "level": assessment.level.value,
            "growth_derivative": assessment.growth_derivative,
            "spread_direction": assessment.spread_direction,
            "active_hazard_count": assessment.active_hazard_count,
        })

    # 1. Warmup verification: frames 0..1 unconfirmed (hits < min_hits), frame 2 confirmed
    assert records[0]["active_hazard_count"] == 0
    assert records[1]["active_hazard_count"] == 0
    assert records[2]["active_hazard_count"] >= 1

    # 2. Camera mode must have transitioned to DYNAMIC early in the sequence
    assert records[1]["mode"] == "DYNAMIC"

    # 3. Steady state verification (frames 8 through 17):
    # Once derivative lookback window and hysteresis settle, verify every single frame
    for r in records[8:]:
        frame_id = r["frame"]
        assert r["mode"] == "DYNAMIC", f"Frame {frame_id} failed dynamic mode assert"
        assert r["active_hazard_count"] >= 1, f"Frame {frame_id} dropped active hazard"
        assert abs(r["growth_derivative"]) < 0.10, (
            f"Frame {frame_id} had unexpected growth derivative: {r['growth_derivative']} (expected near 0.0)"
        )
        assert r["level"] == "MONITORING", (
            f"Frame {frame_id} risk level was {r['level']}, expected MONITORING"
        )
        assert r["spread_direction"] in ("Stable / Stationary", "Static"), (
            f"Frame {frame_id} spread direction was {r['spread_direction']}, expected Stable / Stationary"
        )
