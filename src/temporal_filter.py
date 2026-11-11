"""Temporal Filtering and Optical Flow Verification for False Positive Reduction.

Eliminates industrial artifacts such as fog, steam, transient light reflections,
and camera sensor noise by enforcing temporal persistence and upward/expansion
motion dynamics (buoyancy characteristics of real smoke).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedDetection:
    """Represents a temporally tracked detection across sequential frames.

    Attributes:
        history_boxes: Sliding window of the last `queue_size` bounding boxes
            for this track. Used to compute spatial expansion trends.
    """

    track_id: int
    class_name: str
    last_box: Tuple[float, float, float, float]
    history_boxes: deque  # deque(maxlen=queue_size) of (x1,y1,x2,y2) tuples
    hits: int
    age: int
    misses: int
    confirmed: bool = False
    motion_score: float = 0.0

    @property
    def expansion_trend(self) -> float:
        """Returns the relative area change over the history window.

        Positive → detection is growing (expanding smoke/fire).
        Negative → detection is shrinking.
        Returns 0.0 if fewer than 2 samples are available.
        """
        if len(self.history_boxes) < 2:
            return 0.0
        def _area(box: Tuple[float, float, float, float]) -> float:
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        oldest_area = _area(self.history_boxes[0])
        newest_area = _area(self.history_boxes[-1])
        if oldest_area <= 0:
            return 0.0
        return (newest_area - oldest_area) / oldest_area



def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0 else 0.0


class TemporalFilter:
    """Reduces false alarms using temporal sliding windows and Gunnar-Farneback optical flow.

    Optical flow modes:
      - 'roi'  (default): Farneback flow computed only within each detection bounding box.
                          Most CPU-efficient; scales with number of detections.
      - 'full': Farneback flow computed on the entire frame; best for dense detections
                or when global motion context is needed.
    """

    def __init__(
        self,
        queue_size: int = 15,
        min_hits: int = 4,
        max_misses: int = 5,
        iou_match_threshold: float = 0.25,
        optical_flow_enabled: bool = True,
        optical_flow_mode: str = "roi",
        min_motion_magnitude: float = 0.5,
        vertical_bias_weight: float = 0.6,
        radial_expansion_weight: float = 0.4,
    ):
        self.queue_size = queue_size
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.iou_match_threshold = iou_match_threshold
        self.optical_flow_enabled = optical_flow_enabled
        self.optical_flow_mode = (
            optical_flow_mode.lower() if isinstance(optical_flow_mode, str) else "roi"
        )
        if self.optical_flow_mode not in ("roi", "full"):
            logger.warning(
                "Invalid optical_flow_mode '%s'. Valid options: 'roi', 'full'. Falling back to 'roi'.",
                optical_flow_mode,
            )
            self.optical_flow_mode = "roi"
        self.min_motion_magnitude = min_motion_magnitude
        self.vertical_bias_weight = vertical_bias_weight
        self.radial_expansion_weight = radial_expansion_weight

        self.tracks: Dict[int, TrackedDetection] = {}
        self.next_track_id: int = 1

        self.prev_gray_frame: Optional[np.ndarray] = None
        self.current_flow: Optional[np.ndarray] = None


    def update(
        self,
        frame: np.ndarray,
        raw_detections: Optional[List[Detection]] = None,
        detections: Optional[List[Detection]] = None,
        M: Optional[np.ndarray] = None,
        camera_mode: str = "static",
        is_transitioning: bool = False,
    ) -> List[Detection]:
        """Filters raw detections against temporal queue, optical flow dynamics, and ego-motion.

        Args:
            frame: Current BGR image frame.
            raw_detections: List of Detection instances output by detector.
            detections: Alias for raw_detections.
            M: Optional 2x3 Affine transformation matrix from previous to current frame (ego-motion).
            camera_mode: "static" or "dynamic".
            is_transitioning: True if camera mode changed on this frame.
        """
        active_raw = raw_detections if raw_detections is not None else (detections or [])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 0. Camera mode baseline handling:
        # In STATIC mode (stationary industrial CCTV), camera is stationary baseline; M is ignored (treated as Identity I)
        # to prevent residual RANSAC sub-pixel jitter from drifting boxes or warping frames.
        # Ego-motion compensation is only applied when camera_mode is DYNAMIC.
        is_dynamic = (str(camera_mode).lower() == "dynamic")
        apply_motion = (
            is_dynamic
            and (M is not None)
            and not np.allclose(M, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), atol=1e-3)
        )

        if apply_motion and self.tracks:
            for track in self.tracks.values():
                track.last_box = self._transform_box(track.last_box, M, w, h)
                new_hist = deque(maxlen=self.queue_size)
                for b in track.history_boxes:
                    new_hist.append(self._transform_box(b, M, w, h))
                track.history_boxes = new_hist

        # 1. Optical flow handling based on mode
        # Warp prev_gray_frame with M only in DYNAMIC mode to eliminate background camera motion.
        effective_prev_gray = self.prev_gray_frame
        if apply_motion and self.prev_gray_frame is not None:
            effective_prev_gray = cv2.warpAffine(
                self.prev_gray_frame,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        if self.optical_flow_mode == "full":
            if self.optical_flow_enabled and effective_prev_gray is not None and not is_transitioning:
                self.current_flow = cv2.calcOpticalFlowFarneback(
                    prev=effective_prev_gray,
                    next=gray,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
            else:
                self.current_flow = None
        else:
            # 'roi' mode: flow computed per-detection box inside _evaluate_motion
            self.current_flow = None

        # 2. Associate incoming detections with existing tracks
        matched_tracks, unmatched_detections = self._associate_detections(active_raw)

        # 3. Update matched tracks
        for track_id, det_idx in matched_tracks.items():
            track = self.tracks[track_id]
            det = active_raw[det_idx]

            track.last_box = det.box
            track.history_boxes.append(det.box)
            track.hits += 1
            track.age += 1
            track.misses = 0

            # Evaluate motion dynamics on smoke
            motion_score = 0.5 if is_transitioning else self._evaluate_motion(det.box, gray, prev_gray=effective_prev_gray)
            track.motion_score = motion_score

            if track.hits >= self.min_hits:
                track.confirmed = True

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = active_raw[det_idx]
            history = deque(maxlen=self.queue_size)
            history.append(det.box)

            motion_score = 0.5 if is_transitioning else self._evaluate_motion(det.box, gray, prev_gray=effective_prev_gray)

            new_track = TrackedDetection(
                track_id=self.next_track_id,
                class_name=det.class_name,
                last_box=det.box,
                history_boxes=history,
                hits=1,
                age=1,
                misses=0,
                confirmed=(self.min_hits <= 1),
                motion_score=motion_score,
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # 5. Age out tracks that were not matched in this frame
        all_matched_ids = set(matched_tracks.keys())
        stale_ids = []
        for t_id, track in self.tracks.items():
            if t_id not in all_matched_ids:
                track.misses += 1
                track.age += 1
                if track.misses > self.max_misses:
                    stale_ids.append(t_id)

        for s_id in stale_ids:
            del self.tracks[s_id]

        # 6. Store current frame as previous frame for next iteration
        self.prev_gray_frame = gray.copy()

        # 7. Filter and return only temporally verified detections
        verified_detections: List[Detection] = []
        for det in active_raw:
            matched_id = None
            for t_id, d_idx in matched_tracks.items():
                if active_raw[d_idx] is det:
                    matched_id = t_id
                    break

            if matched_id is not None:
                track = self.tracks[matched_id]
                # If optical flow is active and it's smoke, ensure motion score is not strictly zero/negative
                flow_passed = True
                if self.optical_flow_enabled and det.class_name.lower() == "smoke":
                    if track.motion_score < 0.1 and track.hits >= self.min_hits:
                        # Suspect static noise (painted cloud or static reflection)
                        flow_passed = False

                if track.confirmed and flow_passed:
                    det.extra["track_id"] = track.track_id
                    det.extra["hits"] = track.hits
                    det.extra["motion_score"] = round(track.motion_score, 3)
                    det.extra["expansion_trend"] = round(track.expansion_trend, 4)
                    verified_detections.append(det)


        return verified_detections

    def _associate_detections(
        self,
        raw_detections: List[Detection],
    ) -> Tuple[Dict[int, int], List[int]]:
        """Matches detections to existing tracks via highest IoU or normalized centroid proximity."""
        matched: Dict[int, int] = {}
        unmatched_dets = list(range(len(raw_detections)))

        if not self.tracks or not raw_detections:
            return matched, unmatched_dets

        track_ids = list(self.tracks.keys())
        for d_idx, det in enumerate(raw_detections):
            best_score = self.iou_match_threshold - 1e-6
            best_track_id = None

            det_cx, det_cy = det.center
            det_diag = max(1.0, (det.width**2 + det.height**2)**0.5)

            best_centroid_dist = float("inf")
            for t_id in track_ids:
                if t_id in matched:
                    continue
                track = self.tracks[t_id]
                if track.class_name.lower() != det.class_name.lower():
                    continue

                iou = compute_iou(det.box, track.last_box)
                if iou >= self.iou_match_threshold:
                    if iou > best_score:
                        best_score = iou
                        best_track_id = t_id
                elif best_score < self.iou_match_threshold:
                    # Centroid distance fallback for deforming/expanding smoke
                    t_x1, t_y1, t_x2, t_y2 = track.last_box
                    t_cx, t_cy = (t_x1 + t_x2) / 2.0, (t_y1 + t_y2) / 2.0
                    dist = ((det_cx - t_cx)**2 + (det_cy - t_cy)**2)**0.5
                    # Select the closest track within 35% bounding box diagonal
                    if dist <= (0.35 * det_diag) and dist < best_centroid_dist:
                        best_centroid_dist = dist
                        best_track_id = t_id

            if best_track_id is not None:
                matched[best_track_id] = d_idx
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)

        return matched, unmatched_dets


    def _transform_box(
        self,
        box: Tuple[float, float, float, float],
        M: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        """Transforms bounding box [x1, y1, x2, y2] using 2x3 affine matrix M."""
        x1, y1, x2, y2 = box
        pts = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ], dtype=np.float32).T  # 3x4

        warped = M @ pts  # 2x4
        wx1 = float(np.min(warped[0, :]))
        wy1 = float(np.min(warped[1, :]))
        wx2 = float(np.max(warped[0, :]))
        wy2 = float(np.max(warped[1, :]))

        # Clamp to frame boundary
        wx1 = max(0.0, min(float(frame_w - 1), wx1))
        wy1 = max(0.0, min(float(frame_h - 1), wy1))
        wx2 = max(wx1 + 1.0, min(float(frame_w), wx2))
        wy2 = max(wy1 + 1.0, min(float(frame_h), wy2))
        return (wx1, wy1, wx2, wy2)

    def _evaluate_motion(
        self,
        box: Tuple[float, float, float, float],
        current_gray: Optional[np.ndarray] = None,
        prev_gray: Optional[np.ndarray] = None,
    ) -> float:
        """Analyzes optical flow inside ROI for characteristic smoke upward and expansion motion."""
        if not self.optical_flow_enabled:
            return 1.0

        effective_prev = prev_gray if prev_gray is not None else self.prev_gray_frame

        if self.optical_flow_mode == "full":
            if self.current_flow is None:
                return 1.0  # Pass if flow is not computed yet

            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = self.current_flow.shape[:2]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            flow_roi = self.current_flow[y1:y2, x1:x2]
        else:
            # Per-ROI Gunnar-Farneback optical flow computation: O(Σ(wi×hi))
            if current_gray is None or effective_prev is None:
                return 1.0

            if current_gray.ndim == 3:
                current_gray = cv2.cvtColor(current_gray, cv2.COLOR_BGR2GRAY)

            if effective_prev.shape != current_gray.shape:
                return 1.0

            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = current_gray.shape[:2]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Farneback minimum ROI protection: if ROI is too small (< winsize=15), flow is unreliable
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                return 0.5  # Neutral fallback for tiny far-away regions

            prev_roi = effective_prev[y1:y2, x1:x2]
            curr_roi = current_gray[y1:y2, x1:x2]

            if prev_roi.size == 0 or curr_roi.size == 0:
                return 1.0

            flow_roi = cv2.calcOpticalFlowFarneback(
                prev=prev_roi,
                next=curr_roi,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        if flow_roi.size == 0:
            return 1.0

        fx = flow_roi[:, :, 0]
        fy = flow_roi[:, :, 1]

        magnitude, _ = cv2.cartToPolar(fx, fy)
        mean_mag = float(np.mean(magnitude))

        if mean_mag < self.min_motion_magnitude:
            # Completely static region (e.g. painted smoke / cloud)
            return 0.05

        # In image coordinates, y increases downward.
        # Upward motion means fy < 0.
        upward_motion = float(np.mean(-fy))
        # Clear upward motion gets 1.0; downward motion (upward_motion < -0.1) gets severe penalty 0.1; neutral/slight drift gets 0.4
        if upward_motion > 0.1:
            vertical_score = 1.0
        elif upward_motion < -0.1:
            vertical_score = 0.1
        else:
            vertical_score = 0.4

        # Radial divergence / outward spread estimation
        # Compare velocity vectors to position relative to box center
        cx = (x2 - x1) / 2.0
        cy = (y2 - y1) / 2.0
        grid_y, grid_x = np.mgrid[0 : (y2 - y1), 0 : (x2 - x1)]
        rel_x = grid_x - cx
        rel_y = grid_y - cy

        # Dot product with radial direction
        radial_alignment = (fx * rel_x + fy * rel_y) / (np.sqrt(rel_x**2 + rel_y**2) + 1e-5)
        mean_rad = float(np.mean(radial_alignment))
        if mean_rad > 0.05:
            expansion_score = 1.0
        elif mean_rad < -0.05:
            expansion_score = 0.1
        else:
            expansion_score = 0.5

        final_motion_score = (
            self.vertical_bias_weight * vertical_score + self.radial_expansion_weight * expansion_score
        )
        return float(final_motion_score)


    def reset(self) -> None:
        """Resets filter memory and tracked objects."""
        self.tracks.clear()
        self.next_track_id = 1
        self.prev_gray_frame = None
        self.current_flow = None

