"""Smoke and Fire Detection Inference Engine.

Supports both PyTorch (.pt) and ONNX Runtime (.onnx) backends with automatic
device acceleration (CUDA, MPS, CPU) and telemetry latency tracking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detected fire/smoke bounding box and metadata.

    Note: The pipeline currently implements Object Detection via 2D bounding boxes.
    The `mask` attribute is reserved for future instance segmentation (YOLO-seg) compatibility.
    """

    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    area: float
    mask: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates centroid (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


@dataclass
class FrameDetections:
    """Container for all detections in a single frame along with latency metrics."""

    detections: List[Detection]
    inference_time_ms: float
    fps: float
    frame_width: int
    frame_height: int
    total_smoke_area: float = 0.0
    total_fire_area: float = 0.0

    def __post_init__(self):
        for det in self.detections:
            if det.class_name.lower() == "smoke":
                self.total_smoke_area += det.area
            elif det.class_name.lower() == "fire":
                self.total_fire_area += det.area


class SmokeFireDetector:
    """Unified detector supporting YOLO PyTorch weights (.pt) and ONNX (.onnx)."""

    DEFAULT_CLASSES = {0: "smoke", 1: "fire"}

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
        class_mapping: Optional[Dict[int, str]] = None,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_mapping = class_mapping or self.DEFAULT_CLASSES

        self.device = self._resolve_device(device)
        self.is_onnx = self.model_path.suffix.lower() == ".onnx"

        self.model = None
        self.ort_session = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolves target compute device: CUDA, Apple MPS, or CPU."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Initializes either YOLO PyTorch or ONNX Runtime session."""
        if self.is_onnx:
            if ort is None:
                raise ImportError("onnxruntime is required to run .onnx models.")
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            logger.info("Loading ONNX session from %s with providers %s", self.model_path, providers)
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
            self.ort_session = ort.InferenceSession(
                str(self.model_path), sess_options=sess_options, providers=providers
            )
        else:
            if YOLO is None:
                raise ImportError("ultralytics package is required to run .pt models.")
            logger.info("Loading YOLO PyTorch model from %s on device %s", self.model_path, self.device)
            self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """Performs smoke & fire detection on a single frame (BGR numpy array)."""
        height, width = frame.shape[:2]
        start_time = time.perf_counter()

        detections: List[Detection] = []

        if self.is_onnx:
            detections = self._infer_onnx(frame)
        else:
            detections = self._infer_pytorch(frame)

        elapsed = time.perf_counter() - start_time
        inference_time_ms = elapsed * 1000.0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return FrameDetections(
            detections=detections,
            inference_time_ms=inference_time_ms,
            fps=fps,
            frame_width=width,
            frame_height=height,
        )

    def _infer_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """Runs Ultralytics YOLO inference."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        masks_data = None
        if hasattr(res, "masks") and res.masks is not None:
            masks_data = res.masks.data.cpu().numpy()

        for idx, (box, conf, cls_id) in enumerate(zip(coords, confs, classes)):
            x1, y1, x2, y2 = [float(c) for c in box]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            cls_name = self.class_mapping.get(cls_id, self.model.names.get(cls_id, f"class_{cls_id}"))

            mask = None
            if masks_data is not None and idx < len(masks_data):
                mask = masks_data[idx]

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=str(cls_name),
                    area=float(area),
                    mask=mask,
                )
            )

        return detections

    def _letterbox(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes frame to self.img_size while preserving aspect ratio,
        padding remaining area with gray (114, 114, 114) — matching
        Ultralytics convention.
        
        Returns:
            letterboxed: The padded square image
            scale: The scale factor applied
            pad: (pad_w, pad_h) pixel padding offsets
        """
        h, w = frame.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        dw = (self.img_size - new_w) / 2.0
        dh = (self.img_size - new_h) / 2.0
        top = int(round(dh - 0.1))
        bottom = max(0, self.img_size - new_h - top)
        left = int(round(dw - 0.1))
        right = max(0, self.img_size - new_w - left)
        top = max(0, top)
        left = max(0, left)

        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return letterboxed, scale, (left, top)


    def _infer_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Preprocesses frame, executes ONNX Runtime session, and postprocesses."""
        # Letterbox/Resize to square
        letterboxed, scale, (pad_w, pad_h) = self._letterbox(frame)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: tensor})
        raw_output = outputs[0]  # shape: [1, num_classes + 4, num_anchors] or [1, num_anchors, ...]

        if raw_output.shape[1] < raw_output.shape[2]:
            raw_output = np.transpose(raw_output, (0, 2, 1))

        predictions = raw_output[0]  # [num_anchors, 4 + classes]
        boxes_xywh = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        # Convert xywh to xyxy on original scale
        raw_x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        raw_y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        raw_x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        raw_y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        x1 = (raw_x1 - pad_w) / scale
        y1 = (raw_y1 - pad_h) / scale
        x2 = (raw_x2 - pad_w) / scale
        y2 = (raw_y2 - pad_h) / scale

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # cv2.dnn.NMSBoxes expects [x, y, w, h] format (top-left corner + dimensions)
        nms_w = x2 - x1
        nms_h = y2 - y1
        boxes_xywh_for_nms = np.column_stack([x1, y1, nms_w, nms_h])

        # NMS via OpenCV
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh_for_nms.tolist(),
            scores=confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections: List[Detection] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                box = tuple(boxes_xyxy[idx].tolist())
                conf = float(confidences[idx])
                cls_id = int(class_ids[idx])
                cls_name = self.class_mapping.get(cls_id, f"class_{cls_id}")
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

                detections.append(
                    Detection(
                        box=box,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        area=area,
                    )
                )

        return detections

    def export_onnx(self, output_path: Optional[Union[str, Path]] = None, half: bool = False) -> Path:
        """Exports underlying PyTorch model to ONNX for edge inference."""
        if self.is_onnx:
            logger.info("Model is already in ONNX format.")
            return self.model_path

        if self.model is None:
            raise ValueError("No PyTorch model loaded to export.")

        logger.info("Exporting YOLO model to ONNX format (half=%s)...", half)
        export_path = self.model.export(format="onnx", half=half, imgsz=self.img_size)
        target = Path(export_path)

        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            target.rename(dest)
            target = dest

        logger.info("Model successfully exported to %s", target)
        return target
