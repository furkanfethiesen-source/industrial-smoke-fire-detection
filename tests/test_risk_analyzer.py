"""Unit tests for Dynamic Risk Analyzer and Growth Derivative (dA/dt)."""

import pytest

from src.detector import Detection
from src.risk_analyzer import RiskAnalyzer, RiskLevel


def test_risk_analyzer_normal_when_empty():
    analyzer = RiskAnalyzer()
    assessment = analyzer.analyze(detections=[], frame_width=640, frame_height=480, current_time=1.0)
    assert assessment.level == RiskLevel.NORMAL
    assert assessment.total_area_ratio == 0.0
    assert assessment.growth_derivative == 0.0


def test_risk_analyzer_monitoring_on_static_smoke():
    """Statik duman (sabit alan) için MONITORING beklenmeli, türev ~0 olmalı."""
    analyzer = RiskAnalyzer()
    # 2000 pixels on 640x480 (~307,200 total) → ~0.65% area
    det = Detection(box=(50.0, 50.0, 90.0, 100.0), confidence=0.8, class_id=0, class_name="smoke", area=2000.0)

    # Frame 1
    analyzer.analyze([det], frame_width=640, frame_height=480, current_time=1.0)
    # Frame 2 (after 1 sec, same area → dA/dt = 0 with both absolute and relative formulas)
    a2 = analyzer.analyze([det], frame_width=640, frame_height=480, current_time=2.0)

    assert a2.level == RiskLevel.MONITORING
    assert abs(a2.growth_derivative) < 0.01



def test_risk_analyzer_escalates_to_warning_and_critical():
    """dA/dt bağıl büyüme formülü ile MONITORING → WARNING → CRITICAL kademeli artışını test eder.

    Bağıl formül: growth_derivative = (A_now - A_past) / A_past / dt
    Eşikler:
      warning_growth_rate = 0.30 (saniyede %30 göreceli büyüme)
      critical_growth_rate = 1.50 (saniyede %150 göreceli büyüme / hızlı yayılma)
      critical_area_ratio = 0.50 (alan eşiğini devre dışı bırakıp salt türevi sınamak için yüksek tutuldu)

    Beklenen Davranış:
      - Frame 1 (t=1.0s): Başlangıç alanı %1.0 (10.000 px).
        Geçmiş türev verisi olmadığı için durum MONITORING olmalıdır.
      - Frame 2 (t=2.0s): Alan %1.5'e çıkar (15.000 px).
        growth_derivative = (0.015 - 0.010) / 0.010 / 1.0 = +0.50 /s.
        0.50 >= warning_growth_rate (0.30) ve < critical_growth_rate (1.50).
        Risk seviyesi tam olarak WARNING olmalıdır.
      - Frame 3 (t=3.0s): Alan %4.5'e fırlar (45.000 px).
        growth_derivative = (0.045 - 0.015) / 0.015 / 1.0 = +2.00 /s.
        2.00 >= critical_growth_rate (1.50).
        Risk seviyesi CRITICAL'a yükselmeli ve uyarı mesajında büyüme patlaması belirtilmelidir.
    """
    analyzer = RiskAnalyzer(
        warning_growth_rate=0.30,
        critical_growth_rate=1.50,
        critical_area_ratio=0.50,
    )
    w, h = 1000, 1000

    # Frame 1: 1% area (10,000 px) -> MONITORING
    det1 = Detection(box=(0, 0, 100, 100), confidence=0.9, class_id=0, class_name="smoke", area=10_000.0)
    a1 = analyzer.analyze([det1], frame_width=w, frame_height=h, current_time=1.0)
    assert a1.level == RiskLevel.MONITORING

    # Frame 2: 1.5% area after 1 s (15,000 px) -> dA/dt = +0.50 /s -> WARNING
    det2 = Detection(box=(0, 0, 122.47, 122.47), confidence=0.9, class_id=0, class_name="smoke", area=15_000.0)
    a2 = analyzer.analyze([det2], frame_width=w, frame_height=h, current_time=2.0)
    assert a2.growth_derivative == pytest.approx(0.50, rel=1e-2)
    assert a2.level == RiskLevel.WARNING
    assert "yayılma eğiliminde" in a2.message

    # Frame 3: 4.5% area after 1 s (45,000 px) -> dA/dt = (0.045 - 0.015) / 0.015 = +2.00 /s -> CRITICAL
    det3 = Detection(box=(0, 0, 212.13, 212.13), confidence=0.9, class_id=0, class_name="smoke", area=45_000.0)
    a3 = analyzer.analyze([det3], frame_width=w, frame_height=h, current_time=3.0)
    assert a3.growth_derivative == pytest.approx(2.00, rel=1e-2)
    assert a3.level == RiskLevel.CRITICAL
    assert "patlaması" in a3.message or "büyüme" in a3.message



def test_dAdt_monotone_growth():
    """Tanım bağımsız monotonluk testi: büyüyen alan, küçülen alandan her zaman daha yüksek türev üretmeli.

    Bu test dA/dt formülünün hangi varyantı kullanıldığından bağımsız olarak
    geçerli olan temel mantıksal özelliği doğrular.
    """
    w, h = 1000, 1000
    # critical_area_ratio=0.99 → alan eşiği hiç tetiklenmiyor; sadece türev monotonluğu test ediliyor
    kwargs = dict(critical_area_ratio=0.99)

    # Büyüyen senaryo: 1% → 20% area
    analyzer_grow = RiskAnalyzer(**kwargs)
    det_start = Detection(box=(0, 0, 100, 100), confidence=0.9, class_id=0, class_name="smoke", area=10_000.0)
    det_large = Detection(box=(0, 0, 447, 447), confidence=0.9, class_id=0, class_name="smoke", area=200_000.0)
    analyzer_grow.analyze([det_start], frame_width=w, frame_height=h, current_time=1.0)
    a_grow = analyzer_grow.analyze([det_large], frame_width=w, frame_height=h, current_time=2.0)

    # Küçülen senaryo: 20% → 1% area (negatif türev bekleniyor)
    analyzer_shrink = RiskAnalyzer(**kwargs)
    analyzer_shrink.analyze([det_large], frame_width=w, frame_height=h, current_time=1.0)
    a_shrink = analyzer_shrink.analyze([det_start], frame_width=w, frame_height=h, current_time=2.0)

    assert a_grow.growth_derivative > a_shrink.growth_derivative, (
        f"Büyüyen alan ({a_grow.growth_derivative:.4f}) küçülen alandan "
        f"({a_shrink.growth_derivative:.4f}) daha yüksek türev üretmeli"
    )
    # Büyüyen alan → WARNING veya CRITICAL beklenmeli (yüksek büyüme)
    assert a_grow.level in (RiskLevel.WARNING, RiskLevel.CRITICAL), (
        f"Büyüyen alan WARNING/CRITICAL olmalı; got {a_grow.level}"
    )
    # Küçülen alan negatif türev → MONITORING veya NORMAL olmalı
    # (level_floor_scores hysteresis nedeniyle CRITICAL kalmaz çünkü sadece 2. kare)
    assert a_shrink.level in (RiskLevel.NORMAL, RiskLevel.MONITORING), (
        f"Küçülen alan NORMAL/MONITORING olmalı (negatif türev); got {a_shrink.level}, "
        f"growth_derivative={a_shrink.growth_derivative:.4f}"
    )




def test_risk_analyzer_hysteresis_cooldown_delay():
    """Alan eşiğiyle tetiklenen CRITICAL'ın cooldown geçince düşürüleceğini test eder."""
    analyzer = RiskAnalyzer(
        critical_area_ratio=0.10,
        cooldown_frames=5,
    )
    w, h = 1000, 1000
    # Big hazard triggers immediate CRITICAL (15% area > critical_area_ratio=10%)
    det_crit = Detection(box=(0, 0, 387, 387), confidence=0.95, class_id=1, class_name="fire", area=150000.0)
    a_crit = analyzer.analyze([det_crit], frame_width=w, frame_height=h, current_time=1.0)
    assert a_crit.level == RiskLevel.CRITICAL


    # Small static hazard afterwards (0.5% area) - should downgrade to MONITORING only after cooldown
    det_small = Detection(box=(0, 0, 70, 70), confidence=0.8, class_id=0, class_name="smoke", area=5000.0)
    
    # Frames 1 to 4: Still in cooldown, should remain CRITICAL
    for frame_idx in range(1, 5):
        a = analyzer.analyze([det_small], frame_width=w, frame_height=h, current_time=2.0 + frame_idx * 0.1)
        assert a.level == RiskLevel.CRITICAL, f"Frame {frame_idx} should remain CRITICAL due to hysteresis"

    # Frame 5: Cooldown frames (5) reached, now allowed to downgrade to MONITORING
    a_down = analyzer.analyze([det_small], frame_width=w, frame_height=h, current_time=2.0 + 5 * 0.1)
    assert a_down.level == RiskLevel.MONITORING


def test_risk_analyzer_level_score_consistency_under_hysteresis():
    analyzer = RiskAnalyzer(
        critical_area_ratio=0.10,
        cooldown_frames=5,
    )
    w, h = 1000, 1000
    # Trigger CRITICAL
    det_crit = Detection(box=(0, 0, 387, 387), confidence=0.95, class_id=1, class_name="fire", area=150000.0)
    analyzer.analyze([det_crit], frame_width=w, frame_height=h, current_time=1.0)

    # Next frame has 0 detections (raw_score is 0.0), but level is maintained at CRITICAL by hysteresis
    a = analyzer.analyze([], frame_width=w, frame_height=h, current_time=1.1)
    assert a.level == RiskLevel.CRITICAL
    # Score should NOT drop to 0.0 while level is CRITICAL; it must be consistent with the level (e.g. >= 0.7)
    assert a.score >= 0.7, f"Score {a.score} is inconsistently low for level {a.level}"


def test_first_detection_not_suppressed():
    """When a hazard suddenly appears from a clean scene, growth derivative shouldn't be suppressed to 0."""
    analyzer = RiskAnalyzer(warning_growth_rate=0.30)
    w, h = 640, 480
    # Clean background
    analyzer.analyze([], frame_width=w, frame_height=h, current_time=0.0)

    # Sudden fire detection (e.g. 5% area)
    det = Detection(box=(100, 100, 250, 250), confidence=0.9, class_id=1, class_name="fire", area=22500.0)
    res = analyzer.analyze([det], frame_width=w, frame_height=h, current_time=1.0)

    # Must escalate to at least WARNING or higher due to sudden growth
    assert res.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
    assert res.growth_derivative >= analyzer.warning_growth_rate


def test_spread_direction_uses_recent_window():
    """Verify that spread direction uses the recent window and doesn't get skewed by ancient drift."""
    analyzer = RiskAnalyzer()
    w, h = 640, 480

    # First 10 frames: drift Eastward (x increases)
    for i in range(10):
        det = Detection(box=(50 + i * 5, 50, 100 + i * 5, 100), confidence=0.9, class_id=0, class_name="smoke", area=2500.0)
        analyzer.analyze([det], frame_width=w, frame_height=h, current_time=1.0 + i * 0.1)

    # Next 10 frames: sharp drift Upward (y decreases)
    for i in range(10):
        det = Detection(box=(100, 50 - i * 4, 150, 100 - i * 4), confidence=0.9, class_id=0, class_name="smoke", area=2500.0)
        res = analyzer.analyze([det], frame_width=w, frame_height=h, current_time=2.0 + i * 0.1)

    # The most recent motion is Upward Expansion
    assert "Upward" in res.spread_direction


def test_union_area_calculation_overlapping_boxes():
    """Two identical overlapping boxes should result in a single box area (union), not double."""
    analyzer = RiskAnalyzer()
    w, h = 640, 480

    # Box 1: (100, 100, 200, 200) -> width 100, height 100, area 10,000
    det1 = Detection(box=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0, class_name="smoke", area=10000.0)
    # Box 2: exactly same coordinates
    det2 = Detection(box=(100.0, 100.0, 200.0, 200.0), confidence=0.85, class_id=0, class_name="smoke", area=10000.0)

    # If simple sum was used: 10,000 + 10,000 = 20,000 px -> ratio = 20,000 / (640*480) = ~0.0651
    # With union calculation: true area = 10,000 px -> ratio = 10,000 / (640*480) = ~0.0326
    res = analyzer.analyze([det1, det2], frame_width=w, frame_height=h, current_time=1.0)

    expected_single_ratio = 10000.0 / (w * h)
    # Allow 1-2% discretization tolerance due to binary mask scaling
    assert res.smoke_area_ratio == pytest.approx(expected_single_ratio, rel=0.05)
    assert res.total_area_ratio == pytest.approx(expected_single_ratio, rel=0.05)
    # Crucially, it must be much closer to 1x area than 2x sum area
    assert res.total_area_ratio < (expected_single_ratio * 1.5)


def test_cumulative_scale_zoom_sequence_constant_hazard():
    """Under continuous camera zoom-in, cumulative scale normalization keeps dA/dt ≈ 0 for static hazard."""
    analyzer = RiskAnalyzer(
        warning_growth_rate=0.30,
        critical_growth_rate=1.50,
        history_window_sec=0.5,
    )
    w, h = 640, 480
    base_size = 80.0  # Initial width and height
    center_x, center_y = 320.0, 240.0
    zoom_per_frame = 1.04  # 4% zoom in per frame

    cum_zoom = 1.0
    # Warm up initial frame at t = 0.0
    half_size = (base_size * cum_zoom) / 2.0
    det0 = Detection(
        box=(center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size),
        confidence=0.9,
        class_id=0,
        class_name="smoke",
        area=(base_size * cum_zoom) ** 2,
    )
    res0 = analyzer.analyze([det0], frame_width=w, frame_height=h, current_time=0.0, camera_scale=1.0)
    initial_normalized_area = res0.total_area_ratio

    # Simulate 10 frames of continuous zoom-in over 1.0 second (t = 0.1, 0.2, ... 1.0)
    last_res = res0
    for i in range(1, 11):
        t = i * 0.1
        cum_zoom *= zoom_per_frame
        half_size = (base_size * cum_zoom) / 2.0
        screen_area = (base_size * cum_zoom) ** 2

        # Raw screen box expands due to optical zoom
        det = Detection(
            box=(center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size),
            confidence=0.9,
            class_id=0,
            class_name="smoke",
            area=screen_area,
        )

        last_res = analyzer.analyze(
            [det],
            frame_width=w,
            frame_height=h,
            current_time=t,
            camera_scale=zoom_per_frame,
            camera_mode="dynamic",
        )

        # Normalized total area ratio should remain close to initial ratio despite raw pixel growth
        assert last_res.total_area_ratio == pytest.approx(initial_normalized_area, rel=0.10)

    # In raw pixel terms, area grew by 1.04^20 = ~2.19x (119% growth), which without
    # cumulative normalization would trigger false critical alarm (dA/dt > 1.5).
    # With cumulative scale normalization, dA/dt must remain close to 0.0 (< warning_growth_rate).
    assert abs(last_res.growth_derivative) < 0.20
    assert last_res.level == RiskLevel.MONITORING


def test_static_dynamic_static_scale_reset():
    """Transitioning DYNAMIC -> STATIC must reset cumulative scale so subsequent static analysis is not skewed."""
    analyzer = RiskAnalyzer(history_window_sec=0.5)
    w, h = 640, 480
    det = Detection(box=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="smoke", area=10000.0)

    # 1. Dynamic phase with zoom: cumulative_scale increases
    for i in range(5):
        analyzer.analyze([det], frame_width=w, frame_height=h, current_time=i * 0.1, camera_scale=1.1, camera_mode="dynamic")
    assert analyzer.cumulative_scale > 1.4

    # 2. Return to STATIC mode: cumulative_scale must reset to 1.0
    res_static = analyzer.analyze([det], frame_width=w, frame_height=h, current_time=0.6, camera_scale=1.0, camera_mode="static")
    assert analyzer.cumulative_scale == 1.0

    # Area ratio in static mode matches raw 10,000 / (640*480)
    expected_ratio = 10000.0 / (w * h)
    assert res_static.total_area_ratio == pytest.approx(expected_ratio, rel=0.05)


def test_scale_deadband_ignores_minor_jitter():
    """Scale jitter < 0.003 must be ignored by deadband to avoid random-walk scale drift."""
    analyzer = RiskAnalyzer()
    w, h = 640, 480
    det = Detection(box=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="smoke", area=10000.0)

    # 50 frames with minor scale noise (e.g. 1.001) in DYNAMIC mode
    for i in range(50):
        analyzer.analyze([det], frame_width=w, frame_height=h, current_time=i * 0.1, camera_scale=1.001, camera_mode="dynamic")

    # Due to deadband, cumulative_scale must remain exactly 1.0
    assert analyzer.cumulative_scale == 1.0


def test_spread_direction_compensated_for_camera_pan():
    """Camera pan in DYNAMIC mode must not create artificial spread directions for stationary smoke."""
    analyzer = RiskAnalyzer()
    w, h = 640, 480

    # Camera pans Eastwards: background image features shift Westwards (tx = -15.0 px/frame).
    # Physically stationary smoke shifts Westwards on screen by -15 px/frame.
    for i in range(10):
        shift = i * 15.0
        det = Detection(
            box=(300.0 - shift, 200.0, 350.0 - shift, 250.0),
            confidence=0.9,
            class_id=0,
            class_name="smoke",
            area=2500.0,
        )
        res = analyzer.analyze(
            [det],
            frame_width=w,
            frame_height=h,
            current_time=i * 0.1,
            camera_mode="dynamic",
            dx=-15.0,
            dy=0.0,
        )

    # In compensated reference frame, smoke centroid has not moved
    assert res.spread_direction in ("Stable / Stationary", "Static")


def test_static_to_dynamic_transition_growth_derivative_no_spike():
    """Mode-independent scale tracking must prevent artificial dA/dt spikes across STATIC->DYNAMIC transitions."""
    analyzer = RiskAnalyzer(
        warning_growth_rate=0.30,
        critical_growth_rate=1.50,
        history_window_sec=0.5,
    )
    w, h = 640, 480
    base_box = [200.0, 150.0, 300.0, 250.0]  # Base area 10000 px^2

    zoom_per_frame = 1.02  # 2% optical zoom per frame
    current_scale = 1.0

    # 1. First 6 frames in STATIC mode while camera begins zooming
    for i in range(6):
        t = 1.0 + i * 0.1
        current_scale = zoom_per_frame ** i
        # Zoomed bounding box on screen
        cx, cy = (base_box[0] + base_box[2]) / 2.0, (base_box[1] + base_box[3]) / 2.0
        bw = (base_box[2] - base_box[0]) * current_scale
        bh = (base_box[3] - base_box[1]) * current_scale
        box = (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)
        det = Detection(
            box=box,
            confidence=0.9,
            class_id=0,
            class_name="smoke",
            area=bw * bh,
        )
        res = analyzer.analyze(
            [det],
            frame_width=w,
            frame_height=h,
            current_time=t,
            camera_scale=zoom_per_frame if i > 0 else 1.0,
            camera_mode="static",
            dx=0.0,
            dy=0.0,
        )

    # 2. Frame 6: Camera mode transitions to DYNAMIC
    t_trans = 1.0 + 6 * 0.1
    current_scale = zoom_per_frame ** 6
    cx, cy = (base_box[0] + base_box[2]) / 2.0, (base_box[1] + base_box[3]) / 2.0
    bw = (base_box[2] - base_box[0]) * current_scale
    bh = (base_box[3] - base_box[1]) * current_scale
    box = (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)
    det = Detection(
        box=box,
        confidence=0.9,
        class_id=0,
        class_name="smoke",
        area=bw * bh,
    )
    res_trans = analyzer.analyze(
        [det],
        frame_width=w,
        frame_height=h,
        current_time=t_trans,
        camera_scale=zoom_per_frame,
        camera_mode="dynamic",
        dx=0.0,
        dy=0.0,
    )

    # Transition frame must NOT spike above warning threshold
    assert abs(res_trans.growth_derivative) < 0.20
    assert res_trans.level in (RiskLevel.NORMAL, RiskLevel.MONITORING)



