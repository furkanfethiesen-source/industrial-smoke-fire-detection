"""Camera Motion Mode Decision Engine with Hysteresis.

Tracks sliding-window ego-motion magnitude to classify camera state into:
- STATIC: Fixed industrial CCTV camera with minor vibration or stationary feed.
- DYNAMIC: Pan-tilt-zoom (PTZ), handheld, drone, or significant camera translation.

Includes hysteresis to prevent rapid mode flickering around classification boundaries.
"""

from __future__ import annotations

import collections
import logging
import math
from enum import Enum
from typing import Optional

from src.ego_motion import EgoMotionResult

logger = logging.getLogger(__name__)


class CameraMode(str, Enum):
    """Operational camera state."""

    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"


class MotionModeDecider:
    """Classifies camera state using sliding-window ego-motion statistics with hysteresis."""

    def __init__(
        self,
        mode_override: Optional[str] = None,
        camera_mode: Optional[str] = None,
        window_size: int = 15,
        dynamic_threshold: float = 2.5,  # Translation magnitude threshold (px) to switch to DYNAMIC
        static_threshold: float = 1.0,  # Translation magnitude threshold (px) to switch to STATIC
    ) -> None:
        """Initializes MotionModeDecider.

        Args:
            mode_override: 'auto' (dynamic inference), 'static' (forced), or 'dynamic' (forced).
            camera_mode: Alias for mode_override.
            window_size: Number of historical frames for moving average statistics.
            dynamic_threshold: Pixel displacement required to trigger STATIC -> DYNAMIC transition.
            static_threshold: Pixel displacement required to trigger DYNAMIC -> STATIC transition.
        """
        selected_mode = camera_mode or mode_override or "auto"
        self.mode_override = selected_mode.lower()
        self.window_size = window_size
        self.dynamic_threshold = dynamic_threshold
        self.static_threshold = static_threshold

        self.current_mode: CameraMode = CameraMode.STATIC
        self.translation_history: collections.deque[float] = collections.deque(maxlen=window_size)
        self.last_transition_frame: Optional[int] = None
        self.frame_count: int = 0

    def decide(self, ego_result: EgoMotionResult) -> CameraMode:
        """Evaluates current ego-motion and returns the camera operational mode.

        Args:
            ego_result: EgoMotionResult from EgoMotionEstimator.

        Returns:
            CameraMode: STATIC or DYNAMIC.
        """
        self.frame_count += 1

        if self.mode_override == "static":
            self.current_mode = CameraMode.STATIC
            return self.current_mode
        elif self.mode_override == "dynamic":
            self.current_mode = CameraMode.DYNAMIC
            return self.current_mode

        # 'auto' mode evaluation
        if ego_result.compensated:
            # Euclidean translation magnitude
            mag = math.sqrt(ego_result.tx * ego_result.tx + ego_result.ty * ego_result.ty)
            # Add rotational penalty: 1 degree approx equivalent to 2px at screen boundaries
            mag += abs(ego_result.rotation_deg) * 2.0
            # Add zoom/scale penalty: 1% zoom change creates ~3px displacement at frame borders
            mag += abs(ego_result.scale - 1.0) * 100.0
            self.translation_history.append(mag)
        else:
            # If not compensated (e.g. untextured background), assume 0 motion
            self.translation_history.append(0.0)

        if len(self.translation_history) == 0:
            return self.current_mode

        avg_mag = sum(self.translation_history) / len(self.translation_history)

        # Hysteresis state machine
        if self.current_mode == CameraMode.STATIC:
            if avg_mag >= self.dynamic_threshold:
                logger.info(
                    "Camera mode transition: STATIC -> DYNAMIC (avg_motion: %.2fpx >= %.2fpx)",
                    avg_mag,
                    self.dynamic_threshold,
                )
                self.current_mode = CameraMode.DYNAMIC
                self.last_transition_frame = self.frame_count
        elif self.current_mode == CameraMode.DYNAMIC:
            if avg_mag <= self.static_threshold:
                logger.info(
                    "Camera mode transition: DYNAMIC -> STATIC (avg_motion: %.2fpx <= %.2fpx)",
                    avg_mag,
                    self.static_threshold,
                )
                self.current_mode = CameraMode.STATIC
                self.last_transition_frame = self.frame_count

        return self.current_mode

    @property
    def is_transitioning(self) -> bool:
        """Returns True if a mode switch occurred on the immediate previous or current frame.

        Note:
            Initially False. Returns True only when a genuine transition between
            STATIC and DYNAMIC occurs. Initial startup frames are not treated as transitions.
        """
        if self.last_transition_frame is None:
            return False
        return (self.frame_count - self.last_transition_frame) <= 1
