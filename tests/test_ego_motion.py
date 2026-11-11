import cv2
import numpy as np
import pytest

from src.ego_motion import EgoMotionEstimator, EgoMotionResult


def create_textured_image(height=480, width=640, seed=42) -> np.ndarray:
    """Creates a high-contrast textured image with distinct feature points."""
    np.random.seed(seed)
    # Background texture
    img = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    # Draw geometric shapes with high contrast corners
    for i in range(20, width - 20, 40):
        for j in range(20, height - 20, 40):
            cv2.rectangle(img, (i, j), (i + 15, j + 15), 255, -1)
            cv2.circle(img, (i + 25, j + 25), 6, 0, -1)
    return img


def test_ego_motion_static_scene():
    estimator = EgoMotionEstimator(min_inlier_ratio=0.4)
    img = create_textured_image()
    # Identical frames (zero camera movement)
    result = estimator.estimate(img, img.copy())
    assert result.compensated is True
    assert pytest.approx(result.dx, abs=0.2) == 0.0
    assert pytest.approx(result.dy, abs=0.2) == 0.0
    assert pytest.approx(result.scale, abs=0.05) == 1.0
    assert pytest.approx(result.rotation_deg, abs=0.2) == 0.0
    assert np.allclose(result.M, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-2)


def test_ego_motion_known_translation():
    estimator = EgoMotionEstimator(min_inlier_ratio=0.3)
    img1 = create_textured_image()
    h, w = img1.shape[:2]

    # Ground truth translation: tx = 8.0, ty = -5.0
    tx, ty = 8.0, -5.0
    M_gt = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    img2 = cv2.warpAffine(img1, M_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = estimator.estimate(img1, img2)
    assert result.compensated is True
    assert pytest.approx(result.dx, abs=0.8) == tx
    assert pytest.approx(result.dy, abs=0.8) == ty
    assert pytest.approx(result.scale, abs=0.05) == 1.0


def test_ego_motion_rotation_and_scale():
    estimator = EgoMotionEstimator(min_inlier_ratio=0.3)
    img1 = create_textured_image()
    h, w = img1.shape[:2]

    angle_deg = 2.0
    scale = 1.02
    center = (w / 2.0, h / 2.0)
    M_gt = cv2.getRotationMatrix2D(center, angle_deg, scale)
    img2 = cv2.warpAffine(img1, M_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = estimator.estimate(img1, img2)
    assert result.compensated is True
    assert pytest.approx(result.scale, abs=0.05) == scale
    assert pytest.approx(abs(result.rotation_deg), abs=0.5) == angle_deg


def test_ego_motion_hazard_masking():
    """Verify that moving content in hazard boxes is masked out."""
    estimator = EgoMotionEstimator(min_inlier_ratio=0.3)
    img1 = create_textured_image()
    h, w = img1.shape[:2]

    # Background is static (M_gt = I)
    img2 = img1.copy()

    # Create a large artificial moving object inside a box
    box = [100, 100, 300, 300]
    # In frame 1: white block
    cv2.rectangle(img1, (100, 100), (300, 300), 255, -1)
    # In frame 2: moved completely / inverted pattern
    cv2.rectangle(img2, (150, 150), (350, 350), 0, -1)

    # With masking of the hazard box, background ego motion is measured as 0
    result = estimator.estimate(img1, img2, hazard_boxes=[box])
    assert result.compensated is True
    assert pytest.approx(result.dx, abs=0.3) == 0.0
    assert pytest.approx(result.dy, abs=0.3) == 0.0


def test_ego_motion_degenerate_blank_frames():
    estimator = EgoMotionEstimator(min_inlier_ratio=0.4)
    blank1 = np.zeros((480, 640), dtype=np.uint8)
    blank2 = np.zeros((480, 640), dtype=np.uint8)

    result = estimator.estimate(blank1, blank2)
    assert result.compensated is False
    assert result.inlier_ratio == 0.0
    assert np.allclose(result.M, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_ego_motion_first_frame():
    estimator = EgoMotionEstimator()
    img = create_textured_image()
    result = estimator.estimate(None, img)
    assert result.compensated is False
    assert np.allclose(result.M, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_ego_motion_combined_full_affine_c2():
    """C2 synthetic validation: combined translation, rotation, and scaling."""
    estimator = EgoMotionEstimator(min_inlier_ratio=0.35)
    img1 = create_textured_image()
    h, w = img1.shape[:2]

    # Simultaneous tx, ty, rotation theta, scale s
    angle_deg = 2.5
    scale = 1.03
    tx = 10.0
    ty = -6.0

    center = (w / 2.0, h / 2.0)
    M_gt = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M_gt[0, 2] += tx
    M_gt[1, 2] += ty

    img2 = cv2.warpAffine(img1, M_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = estimator.estimate(img1, img2)
    assert result.compensated is True
    assert result.inlier_ratio >= 0.35
    assert pytest.approx(result.scale, abs=0.04) == scale
    assert pytest.approx(abs(result.rotation_deg), abs=0.6) == angle_deg
    # Frobenius norm difference between estimated M and ground truth M
    diff_norm = np.linalg.norm(result.M - M_gt)
    assert diff_norm < 2.0, f"M difference norm {diff_norm} exceeds threshold"


def test_ego_motion_mask_timing_prev_and_curr_boxes():
    """Verify that masking covers prev_hazard_boxes on prev_gray so moving hazards don't corrupt RANSAC."""
    estimator = EgoMotionEstimator(min_inlier_ratio=0.3)
    img1 = create_textured_image()
    h, w = img1.shape[:2]

    # Background is static
    img2 = img1.copy()

    # Hazard is in box1 in img1, and moves to box2 in img2
    prev_box = [100.0, 100.0, 220.0, 220.0]
    curr_box = [250.0, 250.0, 370.0, 370.0]

    # Moving high-contrast block
    cv2.rectangle(img1, (100, 100), (220, 220), 255, -1)
    cv2.rectangle(img2, (250, 250), (370, 370), 0, -1)

    result = estimator.estimate(img1, img2, hazard_boxes=[prev_box], curr_hazard_boxes=[curr_box])
    assert result.compensated is True
    # Background motion must remain 0 despite large foreground object translation
    assert pytest.approx(result.dx, abs=0.3) == 0.0
    assert pytest.approx(result.dy, abs=0.3) == 0.0
