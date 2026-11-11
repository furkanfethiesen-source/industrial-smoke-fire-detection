"""Camera Ego-Motion Estimation Module.

Estimates 2D background camera motion (tx, ty, rotation, scale) between consecutive
frames using sparse Lucas-Kanade optical flow on Good Features to Track with RANSAC
affine partial estimation (4 degrees of freedom: translation, rotation, uniform scale).

Active smoke and fire detection boxes are masked out to prevent foreground hazard
motion from biasing background camera motion estimation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EgoMotionResult:
    """Telemetry and transformation parameters estimated from camera ego-motion.

    M konvansiyonu:
    estimate() çıktısındaki (tx, ty), p_curr = M @ p_prev bağıntısına göre arka plan
    piksellerinin görüntü düzlemindeki yer değiştirmesidir (arka plan kayması).
    - Kamera Doğu'ya/sağa pan yaptığında sahne ekranda Batı'ya/sola kayar: tx < 0.
    - Kamera Kuzey'e/yukarı tilt yaptığında sahne ekranda Güney'e/aşağı kayar: ty > 0.
    """

    M: np.ndarray  # 2x3 Affine transformation matrix mapping background pixels from prev to curr frame
    tx: float  # Horizontal background pixel displacement (tx < 0: scene shifts left / camera pans right)
    ty: float  # Vertical background pixel displacement (ty > 0: scene shifts down / camera tilts up)
    rotation_deg: float  # Camera rotation in degrees
    scale: float  # Scale change factor (e.g., zoom)
    inlier_ratio: float  # Proportion of feature points supporting the model
    compensated: bool  # True if valid transformation was reliably found

    @property
    def dx(self) -> float:
        """Alias for horizontal translation tx."""
        return self.tx

    @property
    def dy(self) -> float:
        """Alias for vertical translation ty."""
        return self.ty


class EgoMotionEstimator:
    """Estimates camera ego-motion using affine partial transformation with RANSAC."""

    def __init__(
        self,
        min_inlier_ratio: float = 0.40,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        """Initializes EgoMotionEstimator parameters."""
        self.min_inlier_ratio = min_inlier_ratio
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_reproj_threshold = ransac_reproj_threshold

    def estimate(
        self,
        prev_gray: Optional[np.ndarray],
        curr_gray: Optional[np.ndarray],
        hazard_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        curr_hazard_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> EgoMotionResult:
        """Estimates the affine partial transformation matrix M from prev_gray to curr_gray.

        Args:
            prev_gray: Previous frame in single-channel grayscale (uint8).
            curr_gray: Current frame in single-channel grayscale (uint8).
            hazard_boxes: Bounding boxes [x1, y1, x2, y2] to mask out on prev_gray (e.g. prev_hazard_boxes).
            curr_hazard_boxes: Optional bounding boxes on curr_gray to also mask out.

        Returns:
            EgoMotionResult containing the 2x3 matrix M and associated motion metrics.
        """
        identity_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        if prev_gray is None or curr_gray is None:
            return EgoMotionResult(
                M=identity_M,
                tx=0.0,
                ty=0.0,
                rotation_deg=0.0,
                scale=1.0,
                inlier_ratio=0.0,
                compensated=False,
            )

        h, w = prev_gray.shape[:2]

        # 1. Create static mask excluding hazard regions on previous frame
        mask = np.full((h, w), 255, dtype=np.uint8)
        all_boxes = list(hazard_boxes or []) + list(curr_hazard_boxes or [])
        if all_boxes:
            for box in all_boxes:
                x1 = max(0, min(w - 1, int(box[0])))
                y1 = max(0, min(h - 1, int(box[1])))
                x2 = max(x1 + 1, min(w, int(box[2])))
                y2 = max(y1 + 1, min(h, int(box[3])))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        # 2. Extract reliable corner features from the static background
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )

        if prev_pts is None or len(prev_pts) < 10:
            return EgoMotionResult(
                M=identity_M,
                tx=0.0,
                ty=0.0,
                rotation_deg=0.0,
                scale=1.0,
                inlier_ratio=0.0,
                compensated=False,
            )

        # 3. Track points into current frame via Lucas-Kanade optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]

        if len(valid_prev) < 8:
            return EgoMotionResult(
                M=identity_M,
                tx=0.0,
                ty=0.0,
                rotation_deg=0.0,
                scale=1.0,
                inlier_ratio=0.0,
                compensated=False,
            )

        # 4. Estimate Partial Affine (4 DOF: tx, ty, uniform scale s, rotation theta)
        M, inliers = cv2.estimateAffinePartial2D(
            valid_prev,
            valid_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )

        if M is None or inliers is None:
            return EgoMotionResult(
                M=identity_M,
                tx=0.0,
                ty=0.0,
                rotation_deg=0.0,
                scale=1.0,
                inlier_ratio=0.0,
                compensated=False,
            )

        num_inliers = int(np.count_nonzero(inliers))
        total_pts = len(valid_prev)
        inlier_ratio = float(num_inliers) / float(max(1, total_pts))

        if inlier_ratio < self.min_inlier_ratio or num_inliers < 6:
            # Low confidence; mark as not compensated to prevent erroneous warping
            return EgoMotionResult(
                M=identity_M,
                tx=0.0,
                ty=0.0,
                rotation_deg=0.0,
                scale=1.0,
                inlier_ratio=inlier_ratio,
                compensated=False,
            )

        # Decompose M into tx, ty, scale, rotation
        # M = [[s*cos(theta), -s*sin(theta), tx],
        #      [s*sin(theta),  s*cos(theta), ty]]
        a = float(M[0, 0])
        b = float(M[1, 0])
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        scale = math.sqrt(a * a + b * b)
        theta_rad = math.atan2(b, a)
        rotation_deg = math.degrees(theta_rad)

        return EgoMotionResult(
            M=M.astype(np.float32),
            tx=tx,
            ty=ty,
            rotation_deg=rotation_deg,
            scale=scale,
            inlier_ratio=inlier_ratio,
            compensated=True,
        )
