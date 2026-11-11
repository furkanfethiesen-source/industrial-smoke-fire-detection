"""Dynamic Risk Analyzer and Growth Derivative Engine.

Calculates the first-order derivative of smoke/fire area (dA/dt), spread direction,
and evaluates industrial risk levels (NORMAL -> MONITORING -> WARNING -> CRITICAL).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Categorized industrial hazard severity levels."""

    NORMAL = "NORMAL"            # No smoke or fire detected
    MONITORING = "MONITORING"    # Static or low-growth detection under observation
    WARNING = "WARNING"          # Significant growth rate (dA/dt > warning threshold)
    CRITICAL = "CRITICAL"        # Fast explosion, dual fire+smoke, or large coverage


@dataclass
class RiskAssessment:
    """Comprehensive snapshot of risk metrics for the current frame."""

    level: RiskLevel
    score: float                 # Continuous hazard score 0.0 - 1.0
    total_area_ratio: float      # Detected hazard area / total frame area
    smoke_area_ratio: float
    fire_area_ratio: float
    growth_derivative: float     # dA/dt (% per second)
    spread_direction: str        # e.g., "North-East", "Static", "Upward"
    active_hazard_count: int
    message: str


class RiskAnalyzer:
    """Computes dynamic risk metrics, growth derivatives, and state-machine transitions."""

    def __init__(
        self,
        warning_growth_rate: float = 0.30,    # 30% relative growth / second → WARNING
        critical_growth_rate: float = 1.50,   # 150% relative growth / second → CRITICAL
        critical_area_ratio: float = 0.15,    # 15% of frame covered → immediate CRITICAL
        history_window_sec: float = 1.0,      # Moving window for derivative calculation
        cooldown_frames: int = 30,            # Quiet frames required before downgrade
    ):
        """Initialises the RiskAnalyzer.

        Args:
            warning_growth_rate: Relative area growth rate threshold (fraction/s) that
                triggers WARNING.  Formula: (A_now - A_past) / A_past / dt.
                Example: 0.30 means hazard area growing at 30 % of itself per second.
            critical_growth_rate: Relative growth rate (fraction/s) that triggers
                CRITICAL.  Default 1.50 = 150 % / second (area more than doubles in <1 s).
            critical_area_ratio: Absolute frame-coverage ratio (0-1) that immediately
                triggers CRITICAL regardless of growth rate.
            history_window_sec: Look-back window in seconds for the derivative.
            cooldown_frames: Number of consecutive low-risk frames required to downgrade
                the state-machine level (hysteresis).
        """

        self.warning_growth_rate = warning_growth_rate
        self.critical_growth_rate = critical_growth_rate
        self.critical_area_ratio = critical_area_ratio
        self.history_window_sec = history_window_sec
        self.cooldown_frames = cooldown_frames

        # History buffer: deque of tuples (timestamp, total_area_ratio, centroid)
        self.history: deque = deque(maxlen=60)
        self.current_level: RiskLevel = RiskLevel.NORMAL
        self.cooldown_counter: int = 0
        self.cumulative_scale: float = 1.0
        self.cumulative_dx: float = 0.0
        self.cumulative_dy: float = 0.0
        self.last_camera_mode: Optional[str] = None

    def analyze(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        current_time: Optional[float] = None,
        camera_scale: float = 1.0,
        camera_mode: str = "static",
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> RiskAssessment:
        """Evaluates hazard risk based on verified detections in the current frame.

        Args:
            detections: List of verified detections.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            current_time: Current timestamp (seconds).
            camera_scale: Camera zoom/scale factor s estimated from ego-motion.
            camera_mode: Camera operational state ("static" or "dynamic").
            dx: Horizontal camera ego-motion displacement in pixels.
            dy: Vertical camera ego-motion displacement in pixels.
        """
        now = current_time if current_time is not None else time.time()
        frame_area = float(max(1, frame_width * frame_height))

        smoke_area, fire_area, total_area = self._calculate_union_areas(
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        is_dynamic = (str(camera_mode).lower() == "dynamic")
        mode_str = "dynamic" if is_dynamic else "static"

        if not detections:
            # When scene has no active hazard, reset scale and translation offset
            self.cumulative_scale = 1.0
            self.cumulative_dx = 0.0
            self.cumulative_dy = 0.0
            self.last_camera_mode = mode_str
        else:
            # Re-baseline on DYNAMIC -> STATIC transition:
            # Reset cumulative scale to 1.0 for stationary analysis while rescaling
            # past history area ratios to scale=1.0 so dA/dt does not experience artificial jumps.
            if self.last_camera_mode == "dynamic" and mode_str == "static":
                old_scale_sq = self.cumulative_scale * self.cumulative_scale
                if old_scale_sq > 0.0 and self.history:
                    new_hist = deque(maxlen=self.history.maxlen)
                    for t_hist, a_hist, c_hist in self.history:
                        new_hist.append((t_hist, a_hist * old_scale_sq, c_hist))
                    self.history = new_hist
                self.cumulative_scale = 1.0
                self.cumulative_dx = 0.0
                self.cumulative_dy = 0.0

            self.last_camera_mode = mode_str

            # Mode-independent scale tracking:
            # Continuously track optical zoom scale across both STATIC and DYNAMIC modes
            # with deadband to prevent random-walk drift.
            if abs(camera_scale - 1.0) >= 0.003 and camera_scale > 0.0:
                clamped_frame_scale = max(0.5, min(2.0, float(camera_scale)))
                self.cumulative_scale *= clamped_frame_scale
                self.cumulative_scale = max(0.1, min(10.0, self.cumulative_scale))

            # Translation compensation is tied to dynamic camera ego-motion
            if is_dynamic:
                self.cumulative_dx += float(dx)
                self.cumulative_dy += float(dy)
            else:
                self.cumulative_dx = 0.0
                self.cumulative_dy = 0.0

        # Normalize by cumulative zoom scale squared (S_cum^2)
        scale_sq = self.cumulative_scale * self.cumulative_scale
        smoke_area /= scale_sq
        fire_area /= scale_sq
        total_area /= scale_sq

        smoke_ratio = min(1.0, smoke_area / frame_area)
        fire_ratio = min(1.0, fire_area / frame_area)
        total_ratio = min(1.0, total_area / frame_area)

        # Compute centroid of all active hazards and map to reference coordinates
        # M konvansiyonu: dx, dy arka plan piksel kaymasıdır (p_curr = M @ p_prev).
        # Kamera Doğu'ya döndüğünde ekran Batı'ya (dx < 0) kayar; fiziksel sabit nesnenin
        # ham koordinatı sola kaydığından x_ref = x_raw - cumulative_dx sabit kalır.
        raw_centroid = self._compute_centroid(detections)
        if raw_centroid is not None:
            ref_centroid = (raw_centroid[0] - self.cumulative_dx, raw_centroid[1] - self.cumulative_dy)
        else:
            ref_centroid = None

        # Store in time history
        self.history.append((now, total_ratio, ref_centroid))

        # Calculate growth derivative dA/dt
        growth_derivative = self._calculate_derivative(now)

        # Determine spread direction
        spread_direction = self._determine_spread_direction()

        # State Machine & Level Evaluation
        raw_level, raw_score, msg = self._evaluate_raw_risk(
            total_ratio=total_ratio,
            fire_ratio=fire_ratio,
            smoke_ratio=smoke_ratio,
            growth_derivative=growth_derivative,
            hazard_count=len(detections),
        )

        # Apply hysteresis / cooldown
        final_level = self._apply_hysteresis(raw_level)

        # Align continuous score with held hysteresis level to avoid contradictory telemetry (e.g. CRITICAL with 0.0 score)
        level_floor_scores = {
            RiskLevel.CRITICAL: 0.85,
            RiskLevel.WARNING: 0.60,
            RiskLevel.MONITORING: 0.30,
            RiskLevel.NORMAL: 0.0,
        }
        min_allowed_score = level_floor_scores.get(final_level, 0.0)
        final_score = max(raw_score, min_allowed_score)

        return RiskAssessment(
            level=final_level,
            score=min(1.0, max(0.0, final_score)),
            total_area_ratio=round(total_ratio, 4),
            smoke_area_ratio=round(smoke_ratio, 4),
            fire_area_ratio=round(fire_ratio, 4),
            growth_derivative=round(growth_derivative, 4),
            spread_direction=spread_direction,
            active_hazard_count=len(detections),
            message=msg,
        )

    def _compute_centroid(self, detections: List[Detection]) -> Optional[Tuple[float, float]]:
        """Calculates area-weighted center of all detections."""
        if not detections:
            return None
        total_weight = sum(d.area for d in detections)
        if total_weight <= 0:
            return None

        cx = sum(d.center[0] * d.area for d in detections) / total_weight
        cy = sum(d.center[1] * d.area for d in detections) / total_weight
        return (cx, cy)

    def _calculate_union_areas(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float]:
        """Calculates true union areas (non-overlapping) for smoke, fire, and total hazards.

        Prevents double-counting and artificial inflation when multiple detection bounding
        boxes overlap in dense or spreading hazard zones.
        """
        if not detections:
            return 0.0, 0.0, 0.0

        smoke_dets = [d for d in detections if d.class_name.lower() == "smoke"]
        fire_dets = [d for d in detections if d.class_name.lower() == "fire"]

        # Fast path: if no overlapping hazard boxes exist, sum areas directly
        if len(detections) == 1:
            det = detections[0]
            if det.class_name.lower() == "smoke":
                return det.area, 0.0, det.area
            elif det.class_name.lower() == "fire":
                return 0.0, det.area, det.area
            return 0.0, 0.0, 0.0

        # Construct downsampled binary masks (or full resolution canvas) for exact union pixel counting
        # Downsampling by a factor of 2 or 4 preserves sub-percent accuracy while taking < 0.2ms
        scale = 0.5
        mask_w = max(1, int(frame_width * scale))
        mask_h = max(1, int(frame_height * scale))

        smoke_mask = np.zeros((mask_h, mask_w), dtype=np.uint8) if smoke_dets else None
        fire_mask = np.zeros((mask_h, mask_w), dtype=np.uint8) if fire_dets else None

        if smoke_mask is not None:
            for d in smoke_dets:
                x1 = max(0, min(mask_w - 1, int(d.box[0] * scale)))
                y1 = max(0, min(mask_h - 1, int(d.box[1] * scale)))
                x2 = max(x1 + 1, min(mask_w, int(d.box[2] * scale)))
                y2 = max(y1 + 1, min(mask_h, int(d.box[3] * scale)))
                cv2.rectangle(smoke_mask, (x1, y1), (x2, y2), 1, -1)

        if fire_mask is not None:
            for d in fire_dets:
                x1 = max(0, min(mask_w - 1, int(d.box[0] * scale)))
                y1 = max(0, min(mask_h - 1, int(d.box[1] * scale)))
                x2 = max(x1 + 1, min(mask_w, int(d.box[2] * scale)))
                y2 = max(y1 + 1, min(mask_h, int(d.box[3] * scale)))
                cv2.rectangle(fire_mask, (x1, y1), (x2, y2), 1, -1)

        inv_scale_sq = 1.0 / (scale * scale)
        smoke_union = float(np.count_nonzero(smoke_mask)) * inv_scale_sq if smoke_mask is not None else 0.0
        fire_union = float(np.count_nonzero(fire_mask)) * inv_scale_sq if fire_mask is not None else 0.0

        if smoke_mask is not None and fire_mask is not None:
            total_mask = cv2.bitwise_or(smoke_mask, fire_mask)
            total_union = float(np.count_nonzero(total_mask)) * inv_scale_sq
        elif smoke_mask is not None:
            total_union = smoke_union
        elif fire_mask is not None:
            total_union = fire_union
        else:
            total_union = 0.0

        return smoke_union, fire_union, total_union

    def _calculate_derivative(self, current_time: float) -> float:
        """Computes the relative area growth rate: (A_now - A_past) / A_past / dt.

        Unit: fractional area change per second (relative to past area).

        Example: 0.5 means the hazard area grew by 50% of its previous size per second.

        Design rationale (mutlak değil bağıl formül):
        - A small fire doubling in size is just as alarming as a large fire doubling;
            a relative metric captures this symmetry naturally.
        - The absolute formula (dA/dt = ΔA/dt) would rate a small-fire doubling as
            low-risk if the absolute area is small, even though proportional growth
            signals imminent escalation.
        - Thresholds (warning_growth_rate and critical_growth_rate) represent
            fractional area expansion rates per second for industrial early warning.

        Edge cases:
        - A_past ≈ 0 (first detection): triggers warning escalation for sudden hazard appearance.
        - dt < 0.01 s (clock jitter): returns 0.0.
        """
        if len(self.history) < 2:
            return 0.0

        # Find closest sample at or slightly before (current_time - history_window_sec)
        target_time = current_time - self.history_window_sec
        chosen_sample = None
        for timestamp, area_ratio, _ in reversed(self.history):
            if timestamp <= target_time:
                chosen_sample = (timestamp, area_ratio)
                break

        if chosen_sample is None:
            # All samples are newer than target_time → use oldest available
            chosen_sample = (self.history[0][0], self.history[0][1])

        t_past, area_past = chosen_sample
        dt = current_time - t_past

        if dt <= 0.01:
            return 0.0

        current_area = self.history[-1][1]
        # Relative growth: (A_now - A_past) / A_past / dt
        # Guard against near-zero denominator (new detection appearing from nothing)
        _AREA_EPSILON = 1e-5
        if area_past < _AREA_EPSILON:
            if current_area > _AREA_EPSILON:
                # Sudden hazard appearance from clean background -> alert growth rate
                return self.warning_growth_rate * 2.0
            return 0.0

        return (current_area - area_past) / area_past / dt


    def _determine_spread_direction(self) -> str:
        """Estimates spread direction based on centroid migration vector."""

        if len(self.history) < 5:
            return "Static"

        _SPREAD_WINDOW = 10
        valid_centroids = [c for _, _, c in self.history if c is not None]
        if len(valid_centroids) < 4:
            return "Static"

        # Use recent window to avoid obsolete/stale drift from far past
        recent_centroids = valid_centroids[-_SPREAD_WINDOW:]
        c_past = recent_centroids[0]
        c_now = recent_centroids[-1]

        dx = c_now[0] - c_past[0]
        dy = c_now[1] - c_past[1]  # Inverted y: negative means upwards

        distance = (dx**2 + dy**2) ** 0.5
        if distance < 10.0:
            return "Stable / Stationary"

        if abs(dy) > 1.5 * abs(dx):
            return "Upward Expansion" if dy < 0 else "Downward Spread"
        if abs(dx) > 1.5 * abs(dy):
            return "Eastward Expansion" if dx > 0 else "Westward Expansion"

        vertical = "North" if dy < 0 else "South"
        horizontal = "East" if dx > 0 else "West"
        return f"{vertical}-{horizontal} Spread"

    def _evaluate_raw_risk(
        self,
        total_ratio: float,
        fire_ratio: float,
        smoke_ratio: float,
        growth_derivative: float,
        hazard_count: int,
    ) -> Tuple[RiskLevel, float, str]:
        """Calculates instantaneous risk level and continuous danger score.

        growth_derivative is a relative rate (fraction/s):
        e.g. 0.5 = area grew by 50% of itself in the last second.
        """
        if hazard_count == 0 or total_ratio <= 0.0001:
            return RiskLevel.NORMAL, 0.0, "Sahada duman veya yangın tehdidi yok."

        # Normalized danger score: bounded area contribution + growth derivative contribution
        crit_area = max(self.critical_area_ratio, 1e-5)
        area_score = min(0.6, (fire_ratio / crit_area) * 0.4 + (smoke_ratio / crit_area) * 0.2)
        crit_growth = max(self.critical_growth_rate, 1e-5)
        growth_score = min(0.4, max(0.0, growth_derivative) / crit_growth * 0.4)
        risk_score = min(1.0, max(0.0, 0.1 + area_score + growth_score))

        # Immediate Critical conditions
        if total_ratio >= self.critical_area_ratio:
            return (
                RiskLevel.CRITICAL,
                1.0,
                f"Kritik alan aşıldı! Toplam alan: %{total_ratio*100:.1f}",
            )

        if growth_derivative >= self.critical_growth_rate:
            return (
                RiskLevel.CRITICAL,
                max(0.9, risk_score),
                f"Hızlı büyüme patlaması tespit edildi! dA/dt: {growth_derivative:.2f}×/sn",
            )

        # Simultaneous fire + smoke with meaningful relative growth → Critical
        if fire_ratio > 0.01 and smoke_ratio > 0.02 and growth_derivative > 0.15:
            return (
                RiskLevel.CRITICAL,
                0.95,
                "Alev ve duman eşzamanlı aktif yayılma fazında!",
            )

        # Warning conditions
        if growth_derivative >= self.warning_growth_rate or fire_ratio > 0.02:
            return (
                RiskLevel.WARNING,
                max(0.6, risk_score),
                f"Dikkat: Tehlike yayılma eğiliminde. dA/dt: {growth_derivative:.2f}×/sn",
            )


        # Monitoring condition
        return (
            RiskLevel.MONITORING,
            max(0.3, risk_score),
            "Duman/alev varlığı izleniyor; alan büyümesi kontrol altında.",
        )

    def _apply_hysteresis(self, target_level: RiskLevel) -> RiskLevel:
        """Escalates level immediately, but requires cooldown to downgrade."""
        levels_order = {
            RiskLevel.NORMAL: 0,
            RiskLevel.MONITORING: 1,
            RiskLevel.WARNING: 2,
            RiskLevel.CRITICAL: 3,
        }

        current_val = levels_order[self.current_level]
        target_val = levels_order[target_level]

        if target_val >= current_val:
            # Immediate escalation
            self.current_level = target_level
            self.cooldown_counter = 0
        else:
            # Downgrade requested: check cooldown
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_frames:
                self.current_level = target_level
                self.cooldown_counter = 0

        return self.current_level

    def reset(self) -> None:
        """Resets analyzer history and state."""
        self.history.clear()
        self.current_level = RiskLevel.NORMAL
        self.cooldown_counter = 0
        self.cumulative_scale = 1.0
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.last_camera_mode = None
