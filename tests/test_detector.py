"""Unit tests for Detection dataclasses and detector properties."""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from src.detector import Detection, FrameDetections, SmokeFireDetector


def test_detection_dataclass_properties():
    det = Detection(
        box=(100.0, 150.0, 300.0, 350.0),
        confidence=0.88,
        class_id=0,
        class_name="smoke",
        area=40000.0,
    )

    assert det.center == (200.0, 250.0)
    assert det.width == 200.0
    assert det.height == 200.0


def test_frame_detections_area_aggregation():
    d1 = Detection(box=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="smoke", area=100.0)
    d2 = Detection(box=(0, 0, 20, 20), confidence=0.85, class_id=1, class_name="fire", area=400.0)

    frame_dets = FrameDetections(
        detections=[d1, d2],
        inference_time_ms=12.5,
        fps=80.0,
        frame_width=640,
        frame_height=480,
    )

    assert frame_dets.total_smoke_area == 100.0
    assert frame_dets.total_fire_area == 400.0
    assert len(frame_dets.detections) == 2


@pytest.fixture
def detector():
    """Provides a SmokeFireDetector instance without loading model weights."""
    with patch.object(SmokeFireDetector, "_load_model"):
        return SmokeFireDetector(img_size=640)


def test_letterbox_landscape_image(detector):
    """Test _letterbox with a landscape image (1920x1080) scales and pads correctly."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    letterboxed, scale, pad = detector._letterbox(frame)

    assert letterboxed.shape == (640, 640, 3)
    assert scale == pytest.approx(640.0 / 1920.0)
    assert pad == (0, 140)

    # Verify gray padding (114, 114, 114) at top and bottom borders
    assert np.all(letterboxed[0, 0] == [114, 114, 114])
    assert np.all(letterboxed[139, 0] == [114, 114, 114])
    assert np.all(letterboxed[140, 0] == [0, 0, 0])
    assert np.all(letterboxed[499, 0] == [0, 0, 0])
    assert np.all(letterboxed[500, 0] == [114, 114, 114])
    assert np.all(letterboxed[639, 0] == [114, 114, 114])


def test_letterbox_portrait_image(detector):
    """Test _letterbox with a portrait image (1080x1920) scales and pads correctly."""
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    letterboxed, scale, pad = detector._letterbox(frame)

    assert letterboxed.shape == (640, 640, 3)
    assert scale == pytest.approx(640.0 / 1920.0)
    assert pad == (140, 0)

    # Verify gray padding (114, 114, 114) at left and right borders
    assert np.all(letterboxed[0, 0] == [114, 114, 114])
    assert np.all(letterboxed[0, 139] == [114, 114, 114])
    assert np.all(letterboxed[0, 140] == [0, 0, 0])
    assert np.all(letterboxed[0, 499] == [0, 0, 0])
    assert np.all(letterboxed[0, 500] == [114, 114, 114])
    assert np.all(letterboxed[0, 639] == [114, 114, 114])


def test_letterbox_square_image(detector):
    """Test _letterbox with a square image (640x640) has no padding."""
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    letterboxed, scale, pad = detector._letterbox(frame)

    assert letterboxed.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad == (0, 0)


def test_letterbox_fractional_pixel_consistency(detector):
    """Test that letterbox returned pad matches the actual border offsets applied when fractional pixels occur."""
    # Choose dimensions such that new_h has fractional dw/dh, e.g. 357x640 with img_size=640
    frame = np.zeros((357, 640, 3), dtype=np.uint8)
    letterboxed, scale, (pad_w, pad_h) = detector._letterbox(frame)

    assert letterboxed.shape == (640, 640, 3)
    # The actual image height is round(357 * scale)
    actual_img_h = int(357 * scale)
    expected_top = (640 - actual_img_h) // 2
    # Verify pad_h equals the actual top padding
    assert pad_h == (640 - actual_img_h) // 2 or pad_h == int(round((640 - actual_img_h) / 2 - 0.1))
    # Crucially, pixels above pad_h should be padding (114) and at pad_h should be image content (0)
    assert np.all(letterboxed[pad_h - 1, 320] == [114, 114, 114])
    assert np.all(letterboxed[pad_h, 320] == [0, 0, 0])


def test_infer_onnx_letterbox_end_to_end(detector):
    """Test _infer_onnx end-to-end integration with letterbox and coordinate inversion."""
    detector.is_onnx = True
    detector.ort_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    detector.ort_session.get_inputs.return_value = [mock_input]

    # Landscape frame 1920x1080: scale = 1/3, pad = (0, 140)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # An object in original frame at [300, 300, 600, 600]
    # In letterboxed coords: x1=100, y1=240, x2=200, y2=340
    # Model predicts [cx, cy, w, h] = [150, 290, 100, 100]
    pred = np.zeros((1, 6, 10), dtype=np.float32)
    pred[0, 0, 0] = 150.0  # cx
    pred[0, 1, 0] = 290.0  # cy
    pred[0, 2, 0] = 100.0  # w
    pred[0, 3, 0] = 100.0  # h
    pred[0, 4, 0] = 0.1    # smoke score
    pred[0, 5, 0] = 0.9    # fire score
    detector.ort_session.run.return_value = [pred]

    detections = detector._infer_onnx(frame)
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "fire"
    assert det.confidence == pytest.approx(0.9)
    assert det.box[0] == pytest.approx(300.0)
    assert det.box[1] == pytest.approx(300.0)
    assert det.box[2] == pytest.approx(600.0)
    assert det.box[3] == pytest.approx(600.0)


def test_onnx_nms_uses_xywh_format(detector, monkeypatch):
    """NMS'e xywh formatında kutu gönderildiğini doğrular (1c hata regresyonu).

    cv2.dnn.NMSBoxes [x, y, w, h] formatı bekler.
    xyxy gönderilseydi w ve h çok büyük olurdu (x2-x_origin değil x2-0).
    Bu test, NMS'e giden kutunun w < frame_width ve h < frame_height olduğunu kontrol eder.
    """
    detector.is_onnx = True
    detector.ort_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    detector.ort_session.get_inputs.return_value = [mock_input]

    recorded_bboxes = []
    original_nms = cv2.dnn.NMSBoxes

    def spy_nms(bboxes, scores, score_threshold, nms_threshold):
        recorded_bboxes.extend(bboxes)
        return original_nms(bboxes, scores, score_threshold, nms_threshold)

    monkeypatch.setattr(cv2.dnn, "NMSBoxes", spy_nms)

    # Object at letterboxed coords roughly center of 640x480 frame → original ~200x150 box
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # cx=320, cy=240, w=200, h=150 in letterboxed space
    pred = np.zeros((1, 6, 10), dtype=np.float32)
    pred[0, 0, 0] = 320.0
    pred[0, 1, 0] = 240.0
    pred[0, 2, 0] = 200.0
    pred[0, 3, 0] = 150.0
    pred[0, 4, 0] = 0.9   # smoke score
    pred[0, 5, 0] = 0.1
    detector.ort_session.run.return_value = [pred]

    detector._infer_onnx(frame)

    assert len(recorded_bboxes) == 1, "Exactly one box should reach NMS"
    x_nms, y_nms, w_nms, h_nms = recorded_bboxes[0]
    # In xywh format, w and h are the *dimensions* of the box, not absolute coords.
    # xyxy format would give w_nms = x2 ≈ 420 and h_nms = y2 ≈ 315.
    # xywh gives w_nms = 200, h_nms = 150 (letterboxed box width/height).
    assert w_nms < 640, f"NMS w={w_nms} suspiciously large — xyxy may have been sent instead of xywh"
    assert h_nms < 480, f"NMS h={h_nms} suspiciously large — xyxy may have been sent instead of xywh"
    # Specifically: w and h should be box dimensions, not absolute x2/y2 coords
    assert w_nms < 220 and h_nms < 170, (
        f"Expected w≈200, h≈150 for xywh; got w={w_nms:.1f}, h={h_nms:.1f}"
    )


def test_infer_onnx_nms_eliminates_overlapping_boxes(detector):
    """NMS çakışan (yüksek IoU) iki kutudan sadece birini döndürmeli.

    Bu test, NMS formatı doğruyken gerçekten bastırma yapıldığını doğrular.
    YOLOv8 ONNX formatı: [1, 4+num_classes, num_anchors] → code transposes to [1, num_anchors, 6]
    """
    detector.is_onnx = True
    detector.ort_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    detector.ort_session.get_inputs.return_value = [mock_input]

    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # YOLOv8 ONNX format: [1, 4+num_classes, num_anchors]
    # With 2 classes (smoke, fire) and 2 anchors: shape = [1, 6, 2]
    # Detector code: shape[1]=6 > shape[2]=2 → NO transpose → predictions=[6,2]
    # That means we need shape[1] < shape[2] to get transpose.
    # Use many anchors so shape[1]=6 < shape[2]=num_anchors
    num_anchors = 10
    # Build [1, 6, num_anchors] → code transposes to [1, num_anchors, 6]
    raw = np.zeros((1, 6, num_anchors), dtype=np.float32)

    # Box A (anchor 0): cx=320, cy=320, w=200, h=200, smoke=0.95, fire=0.10
    raw[0, 0, 0] = 320.0; raw[0, 1, 0] = 320.0; raw[0, 2, 0] = 200.0; raw[0, 3, 0] = 200.0
    raw[0, 4, 0] = 0.95; raw[0, 5, 0] = 0.10

    # Box B (anchor 1): cx=322, cy=322, w=200, h=200, smoke=0.75 — should be suppressed
    raw[0, 0, 1] = 322.0; raw[0, 1, 1] = 322.0; raw[0, 2, 1] = 200.0; raw[0, 3, 1] = 200.0
    raw[0, 4, 1] = 0.75; raw[0, 5, 1] = 0.10

    # All other anchors have 0 scores → filtered by conf_threshold
    detector.ort_session.run.return_value = [raw]

    detections = detector._infer_onnx(frame)
    assert len(detections) == 1, (
        f"NMS should suppress one of the two highly-overlapping boxes; got {len(detections)}"
    )
    assert detections[0].confidence == pytest.approx(0.95)


def test_letterbox_exact_square_various_aspect_ratios(detector):
    """Ensure letterbox always produces an exact square image without shape mismatch."""
    test_resolutions = [
        (480, 640),
        (720, 1280),
        (1080, 1920),
        (1920, 1080),
        (357, 640),
        (513, 277),
        (640, 640),
        (120, 400),
    ]
    for h, w in test_resolutions:
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        lb, scale, (pad_w, pad_h) = detector._letterbox(dummy)
        assert lb.shape == (detector.img_size, detector.img_size, 3), (
            f"Resolution {h}x{w} produced invalid letterbox shape {lb.shape}"
        )
        assert pad_w >= 0 and pad_h >= 0

