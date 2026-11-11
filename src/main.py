"""Main Industrial Smoke & Fire Detection and Risk Analytics Pipeline.

Connects video stream (RTSP/video/webcam/images) to the detection engine,
applies temporal optical flow false-positive filtering, calculates dynamic
growth derivatives (dA/dt), renders real-time HUD telemetry, and dispatches
webhooks on critical risk levels.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connector.alert import AlertConnector
from src.detector import Detection, FrameDetections, SmokeFireDetector
from src.ego_motion import EgoMotionEstimator, EgoMotionResult
from src.motion_mode import CameraMode, MotionModeDecider
from src.risk_analyzer import RiskAnalyzer, RiskAssessment, RiskLevel
from src.temporal_filter import TemporalFilter
from src.threaded_stream import ThreadedStream

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("IndustrialPipeline")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Graceful Shutdown: OS Signal Handling (SIGTERM / SIGINT)
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()


def _signal_handler(signum: int, frame: object) -> None:
    """Catches SIGTERM/SIGINT and sets the global shutdown event."""
    sig_name = signal.Signals(signum).name if hasattr(signal.Signals, "name") else str(signum)
    logger.warning("Received %s — initiating graceful shutdown...", sig_name)
    _shutdown_event.set()



def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration yaml file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if not path.exists():
        logger.warning("Config file not found at %s. Using default settings.", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class IndustrialPipeline:
    """Orchestrates end-to-end video stream inference and safety telemetry."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_path_override: Optional[str] = None,
        device_override: Optional[str] = None,
        source_id: str = "industrial_cam_01",
    ):
        self.config = config
        det_cfg = config.get("detector", {})
        temp_cfg = config.get("temporal_filter", {})
        risk_cfg = config.get("risk_analyzer", {})
        alert_cfg = config.get("alert_connector", {})

        model_path = model_path_override or det_cfg.get("model_path", "yolov8n.pt")
        device = device_override or det_cfg.get("device", "auto")

        # 1. Detection Engine
        self.detector = SmokeFireDetector(
            model_path=model_path,
            conf_threshold=det_cfg.get("conf_threshold", 0.25),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            img_size=det_cfg.get("img_size", 640),
            device=device,
            class_mapping=det_cfg.get("classes", None),
        )

        # 2. Temporal & Optical Flow Filter
        self.temporal_filter = TemporalFilter(
            queue_size=temp_cfg.get("queue_size", 15),
            min_hits=temp_cfg.get("min_hits", 4),
            optical_flow_enabled=temp_cfg.get("optical_flow_enabled", True),
            optical_flow_mode=temp_cfg.get("optical_flow_mode", "roi"),
            min_motion_magnitude=temp_cfg.get("min_motion_magnitude", 0.5),
            vertical_bias_weight=temp_cfg.get("vertical_bias_weight", 0.6),
            radial_expansion_weight=temp_cfg.get("radial_expansion_weight", 0.4),
        )

        # 3. Dynamic Risk Analyzer
        self.risk_analyzer = RiskAnalyzer(
            warning_growth_rate=risk_cfg.get("warning_growth_rate", 0.30),
            critical_growth_rate=risk_cfg.get("critical_growth_rate", 1.50),
            critical_area_ratio=risk_cfg.get("critical_area_ratio", 0.15),
            history_window_sec=risk_cfg.get("history_window_sec", risk_cfg.get("derivative_interval_sec", 1.0)),
            cooldown_frames=risk_cfg.get("cooldown_frames", 30),
        )

        # 4. Webhook / Alert Connector
        self.alert_connector = AlertConnector(
            enabled=alert_cfg.get("enabled", True),
            webhook_url=alert_cfg.get("webhook_url", "http://localhost:8080/api/v1/alerts"),
            request_timeout_sec=alert_cfg.get("request_timeout_sec", 3.0),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 10.0),
            simulate_only=alert_cfg.get("simulate_only", True),
            source_id=source_id,
        )

        # 5. Ego-Motion Compensation & Dynamic Camera Mode
        cam_cfg = config.get("camera_motion", {})
        self.ego_motion = EgoMotionEstimator(
            min_inlier_ratio=cam_cfg.get("min_inlier_ratio", 0.40),
        )
        self.motion_mode = MotionModeDecider(
            camera_mode=cam_cfg.get("camera_mode", "auto"),
            window_size=cam_cfg.get("window_size", 15),
            dynamic_threshold=cam_cfg.get("dynamic_threshold", 2.5),
            static_threshold=cam_cfg.get("static_threshold", 1.0),
        )
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.prev_hazard_boxes: List[Tuple[float, float, float, float]] = []

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, RiskAssessment, FrameDetections]:
        """Runs single-frame pipeline: Detect -> Filter -> Analyze Risk -> Dispatch Alert -> Render HUD."""
        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Raw Inference
        raw_results = self.detector.detect(frame)
        curr_hazard_boxes = [d.box for d in raw_results.detections]

        # 2. Ego-Motion Compensation & Camera Mode Decision
        # Mask out previous hazard boxes on prev_gray and current hazard boxes to isolate background
        ego_result = self.ego_motion.estimate(
            prev_gray=self.prev_gray_frame,
            curr_gray=curr_gray,
            hazard_boxes=self.prev_hazard_boxes,
            curr_hazard_boxes=curr_hazard_boxes,
        )
        camera_mode = self.motion_mode.decide(ego_result)

        # 3. Temporal False-Positive Filtering (compensated by M)
        verified_detections = self.temporal_filter.update(
            frame=frame,
            detections=raw_results.detections,
            M=ego_result.M,
            camera_mode=camera_mode.value.lower(),
            is_transitioning=self.motion_mode.is_transitioning,
        )

        # 4. Dynamic Risk Analysis (normalized by camera scale and compensated for translation)
        # In DYNAMIC mode, evaluate translation displacement at optical center (w/2, h/2)
        # to correctly account for combined affine transformations (pan + optical zoom)
        if ego_result.compensated and camera_mode == CameraMode.DYNAMIC:
            pt_center = ego_result.M @ np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float32)
            dx = float(pt_center[0] - (w / 2.0))
            dy = float(pt_center[1] - (h / 2.0))
            camera_scale = ego_result.scale
        elif ego_result.compensated:
            dx = 0.0
            dy = 0.0
            camera_scale = ego_result.scale  # Mode-independent scale tracking
        else:
            dx, dy, camera_scale = 0.0, 0.0, 1.0

        assessment = self.risk_analyzer.analyze(
            detections=verified_detections,
            frame_width=w,
            frame_height=h,
            current_time=timestamp,
            camera_scale=camera_scale,
            camera_mode=camera_mode.value.lower(),
            dx=dx,
            dy=dy,
        )

        # Update previous frame and hazard boxes for next iteration
        self.prev_gray_frame = curr_gray.copy()
        self.prev_hazard_boxes = curr_hazard_boxes

        # 5. Alert Triggering on High Risk
        self.alert_connector.maybe_dispatch(assessment)

        # 6. Telemetry & HUD Rendering
        annotated_frame = self.render_hud(
            frame=frame.copy(),
            raw_detections=raw_results.detections,
            verified_detections=verified_detections,
            assessment=assessment,
            latency_ms=raw_results.inference_time_ms,
            fps=raw_results.fps,
            camera_mode=camera_mode.value,
            ego_result=ego_result,
        )

        return annotated_frame, assessment, raw_results

    def render_hud(
        self,
        frame: np.ndarray,
        raw_detections: List[Detection],
        verified_detections: List[Detection],
        assessment: RiskAssessment,
        latency_ms: float,
        fps: float,
        camera_mode: str = "STATIC",
        ego_result: Optional[EgoMotionResult] = None,
    ) -> np.ndarray:
        """Draws industrial telemetry HUD, bounding boxes, growth derivative and risk banner."""
        h, w = frame.shape[:2]

        # 1. Color Palette based on Risk Level
        level_bgr = {
            RiskLevel.NORMAL: (46, 204, 113),
            RiskLevel.MONITORING: (230, 200, 40),
            RiskLevel.WARNING: (0, 140, 255),
            RiskLevel.CRITICAL: (30, 30, 230),
        }
        status_color = level_bgr.get(assessment.level, (255, 255, 255))


        # 2. Draw Bounding Boxes
        for det in verified_detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            is_fire = det.class_name.lower() == "fire"
            box_color = (0, 69, 255) if is_fire else (180, 180, 180)

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label badge
            label = f"{det.class_name.upper()} {det.confidence:.2f}"
            t_id = det.extra.get("track_id")
            if t_id is not None:
                label += f" [ID:{t_id}]"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Top Banner & Telemetry Box Backgrounds (Single blended overlay for efficiency)
        banner_h = 44
        box_w, box_h = 320, 130
        bx, by = 15, h - box_h - 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Risk Indicator Badge
        badge_w = 200
        cv2.rectangle(frame, (0, 0), (badge_w, banner_h), status_color, -1)
        badge_text = f"RISK: {assessment.level.value}"
        cv2.putText(
            frame,
            badge_text,
            (14, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Metrics text in Top Banner
        raw_count = len(raw_detections)
        verified_count = len(verified_detections)
        filtered_out = max(0, raw_count - verified_count)

        banner_info = (
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | "
            f"dA/dt: {assessment.growth_derivative*100:+.1f}%/s | "
            f"Area: {assessment.total_area_ratio*100:.1f}% | "
            f"Doğrulanmamış: {filtered_out}"
        )
        cv2.putText(
            frame,
            banner_info,
            (badge_w + 20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # 4. Bottom-Left Telemetry Box Border
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), status_color, 1)

        cam_info = f"Camera: {camera_mode}"
        if ego_result and ego_result.compensated and camera_mode == "DYNAMIC":
            cam_info += f" [dx:{ego_result.dx:+.1f}, dy:{ego_result.dy:+.1f}]"

        lines = [
            f"Hazard Spread: {assessment.spread_direction}",
            f"Smoke: {assessment.smoke_area_ratio*100:.1f}% | Fire: {assessment.fire_area_ratio*100:.1f}%",
            f"Threat Score: {assessment.score:.2f} / 1.00",
            f"Active Objects: {assessment.active_hazard_count} verified",
            cam_info,
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (bx + 10, by + 24 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        return frame


def run_pipeline(
    source: str = "0",
    model_path: Optional[str] = None,
    config_path: str = "config/config.yaml",
    save_video: Optional[bool] = None,
    output_dir: Optional[str] = None,
    display: Optional[bool] = None,
    max_frames: Optional[int] = None,
) -> None:
    """Runs video stream reader and pipeline executor.

    Integrates ThreadedStream for zero-lag video capture, asynchronous alert
    dispatching, and graceful shutdown on SIGTERM/SIGINT signals.
    """
    # Register OS signal handlers safely in the running thread (main thread)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, AttributeError) as exc:
        logger.debug("Signal registration skipped: %s", exc)

    # Reset shutdown event in case of re-entry (e.g. tests)
    _shutdown_event.clear()

    cfg = load_config(config_path)
    pipe_cfg = cfg.get("pipeline", {})

    # Resolve settings: CLI overrides > config.yaml > hardcoded defaults
    final_save_video = save_video if save_video is not None else pipe_cfg.get("save_output", True)
    final_output_dir = output_dir if output_dir is not None else pipe_cfg.get("output_dir", "output")
    final_display = display if display is not None else pipe_cfg.get("display", False)
    target_fps = pipe_cfg.get("target_fps", 30)

    pipeline = IndustrialPipeline(config=cfg, model_path_override=model_path)

    # Start alert connector worker thread
    pipeline.alert_connector.start()

    # Check if source is image folder or single image
    source_path = Path(source)
    is_image_mode = False
    image_files = []

    if source_path.exists() and source_path.is_dir():
        is_image_mode = True
        image_files = sorted(
            glob.glob(str(source_path / "*.jpg"))
            + glob.glob(str(source_path / "*.png"))
            + glob.glob(str(source_path / "*.jpeg"))
        )
        logger.info("Found %d images in folder: %s", len(image_files), source_path)
    elif source_path.exists() and source_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        is_image_mode = True
        image_files = [str(source_path)]

    out_writer = None
    stream: Optional[ThreadedStream] = None
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    out_video_path = Path(final_output_dir) / "output_stream.mp4"

    logger.info("Starting Industrial Smoke & Fire Detection Pipeline...")
    logger.info("Source: %s | Mode: %s", source, "Images" if is_image_mode else "Video/Stream")

    frame_count = 0
    start_all = time.time()

    try:
        if is_image_mode:
            for img_path in image_files:
                if _shutdown_event.is_set():
                    logger.info("Shutdown requested — stopping image processing.")
                    break

                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                frame_count += 1
                simulated_timestamp = start_all + (frame_count * (1.0 / target_fps))

                annotated, assessment, _ = pipeline.process_frame(
                    frame=frame,
                    timestamp=simulated_timestamp,
                )

                if final_display:
                    cv2.imshow("Industrial Smoke & Fire Analytics", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if final_save_video:
                    if out_writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_writer = cv2.VideoWriter(str(out_video_path), fourcc, float(target_fps), (w, h))
                    out_writer.write(annotated)

                if max_frames and frame_count >= max_frames:
                    break
        else:
            # Video, RTSP, or Webcam — use ThreadedStream for zero-lag capture
            stream = ThreadedStream(source=source).start()
            fps = stream.fps or float(target_fps)

            while (stream.is_alive() or not stream.is_empty()) and not _shutdown_event.is_set():
                ret, frame, is_new = stream.read_with_status()
                if not ret or frame is None:
                    # Buffer empty or EOF reached
                    if getattr(stream, "is_eof", False) and stream.is_empty():
                        logger.info("Source reached EOF and buffer drained. Concluding video stream.")
                        break
                    if getattr(stream, "has_failed", False):
                        logger.error("Video stream failed permanently.")
                        break
                    time.sleep(0.001)
                    continue

                if not is_new:
                    # Same frame as last time — skip to avoid duplicate processing
                    time.sleep(0.001)
                    continue

                frame_count += 1

                # For file sources, derive timestamp from the actual consumed frame index (last_read_frame_idx)
                # and FPS so that dA/dt is not distorted by reader-ahead queues, processing latency, or system load.
                # For live sources (RTSP, webcam), use wall-clock time.
                if getattr(stream, "is_file_source", False):
                    frame_idx = getattr(stream, "last_read_frame_idx", frame_count)
                    current_timestamp = start_all + (frame_idx / fps)
                else:
                    current_timestamp = time.time()


                annotated, assessment, _ = pipeline.process_frame(
                    frame=frame,
                    timestamp=current_timestamp,
                )

                if final_display:
                    cv2.imshow("Industrial Smoke & Fire Analytics", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if final_save_video:
                    if out_writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
                    out_writer.write(annotated)

                if max_frames and frame_count >= max_frames:
                    break

    finally:
        # 1. Video writer — safe release to prevent corrupted mp4
        if out_writer is not None:
            out_writer.release()
            logger.info("Processed video saved to %s", out_video_path)

        # 2. Threaded stream — stop capture thread and release device
        if stream is not None:
            stream.stop()

        # 3. Alert connector — drain pending alerts and stop worker thread
        pipeline.alert_connector.stop()

        # 4. Display cleanup
        if final_display:
            cv2.destroyAllWindows()

        logger.info("All resources released. Clean shutdown complete.")

    total_time = time.time() - start_all
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    logger.info("Pipeline finished. Processed %d frames in %.2fs (Avg: %.1f FPS)", frame_count, total_time, avg_fps)

    if stream is not None and getattr(stream, "has_failed", False):
        raise RuntimeError(f"Video stream '{source}' aborted due to connection failure or reconnect limit exhaustion.")


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Smoke & Fire Detection System")
    parser.add_argument("--source", type=str, default="data/mini_dfire/images/val", help="Video source (RTSP, webcam id, mp4, or image folder)")
    parser.add_argument("--model", type=str, default=None, help="Path to weights (.pt or .onnx)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=None, help="Display visual window (GUI)")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None, help="Save processed video to output/")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output videos")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for tests)")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(
            source=args.source,
            model_path=args.model,
            config_path=args.config,
            save_video=args.save_video,
            output_dir=args.output_dir,
            display=args.display,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        logger.error("Execution failed: %s", exc, exc_info=True)
        sys.exit(1)
