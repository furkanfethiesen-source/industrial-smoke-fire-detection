"""Unit tests for AlertConnector and cooldown dispatching."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from connector.alert import AlertConnector, AlertPayload
from src.risk_analyzer import RiskAssessment, RiskLevel


def test_alert_connector_suppresses_low_risk():
    connector = AlertConnector(enabled=True, simulate_only=True)
    assessment = RiskAssessment(
        level=RiskLevel.MONITORING,
        score=0.3,
        total_area_ratio=0.01,
        smoke_area_ratio=0.01,
        fire_area_ratio=0.0,
        growth_derivative=0.0,
        spread_direction="Static",
        active_hazard_count=1,
        message="Monitoring test",
    )
    # MONITORING should not trigger alarm
    assert not connector.maybe_dispatch(assessment)


def test_alert_connector_dispatches_critical_and_enforces_cooldown():
    connector = AlertConnector(enabled=True, cooldown_seconds=5.0, simulate_only=True)
    assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.95,
        total_area_ratio=0.25,
        smoke_area_ratio=0.15,
        fire_area_ratio=0.10,
        growth_derivative=0.18,
        spread_direction="Upward",
        active_hazard_count=2,
        message="Critical fire explosion test",
    )

    # First critical triggers alert
    dispatched = connector.maybe_dispatch(assessment)
    assert dispatched is True

    # Immediate second critical should be suppressed by cooldown
    suppressed = connector.maybe_dispatch(assessment)
    assert suppressed is False


def test_worker_thread_lifecycle():
    """Tests worker thread startup, alert processing, and graceful shutdown."""
    connector = AlertConnector(enabled=True, simulate_only=True)
    assert connector._worker is None

    connector.start()
    assert connector._worker is not None
    assert connector._worker.is_alive()
    assert connector._worker.daemon is True

    # Starting when already running should be a no-op
    worker_ref = connector._worker
    connector.start()
    assert connector._worker is worker_ref

    # Dispatch alert and ensure worker processes it from queue
    assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.9,
        total_area_ratio=0.2,
        smoke_area_ratio=0.1,
        fire_area_ratio=0.1,
        growth_derivative=0.1,
        spread_direction="Static",
        active_hazard_count=1,
        message="Lifecycle processing test",
    )
    assert connector.maybe_dispatch(assessment) is True
    connector._queue.join()
    assert connector._queue.empty()

    # Stop worker thread gracefully
    connector.stop(timeout=2.0)
    assert connector._worker is None

    # Stopping again should be safe
    connector.stop(timeout=1.0)
    assert connector._worker is None


def test_queue_overflow_drop_oldest():
    """Tests that a full queue drops the oldest alert to enqueue the newest."""
    connector = AlertConnector(enabled=True, simulate_only=True, max_queue_size=3)

    payloads = [
        AlertPayload(
            event_id=f"ALERT-{i}",
            timestamp="2026-10-20T12:00:00Z",
            source_id="camera_01",
            risk_level="CRITICAL",
            risk_score=0.9,
            total_area_ratio=0.2,
            growth_derivative=0.1,
            spread_direction="Static",
            hazard_count=1,
            message=f"Alert payload {i}",
        )
        for i in range(1, 5)
    ]

    # Enqueue 4 items into a queue with capacity 3
    for p in payloads:
        enqueued = connector._enqueue(p)
        assert enqueued is True

    assert connector._queue.qsize() == 3

    # ALERT-1 should have been dropped; remaining should be ALERT-2, ALERT-3, ALERT-4
    queued_events = []
    while not connector._queue.empty():
        item = connector._queue.get_nowait()
        queued_events.append(item.event_id)

    assert queued_events == ["ALERT-2", "ALERT-3", "ALERT-4"]


def test_retry_mechanism_success_after_failure():
    """Tests retry mechanism with exponential backoff recovering on HTTP 200."""
    connector = AlertConnector(
        enabled=True,
        simulate_only=False,
        webhook_url="http://localhost:8080/api/v1/alerts",
        max_retries=3,
    )
    payload = AlertPayload(
        event_id="ALERT-RETRY-01",
        timestamp="2026-10-20T12:00:00Z",
        source_id="camera_01",
        risk_level="CRITICAL",
        risk_score=0.9,
        total_area_ratio=0.2,
        growth_derivative=0.1,
        spread_direction="Static",
        hazard_count=1,
        message="Retry recovery test",
    )

    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Internal Server Error"

    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.text = '{"status": "ok"}'

    with patch("connector.alert.requests.post", side_effect=[mock_fail, mock_fail, mock_ok]) as mock_post, \
         patch("connector.alert.time.sleep") as mock_sleep:
        success = connector._do_send(payload)

        assert success is True
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)


def test_retry_mechanism_exhausted_failures():
    """Tests retry mechanism returns False when all retry attempts fail."""
    connector = AlertConnector(
        enabled=True,
        simulate_only=False,
        webhook_url="http://localhost:8080/api/v1/alerts",
        max_retries=3,
    )
    payload = AlertPayload(
        event_id="ALERT-FAIL-01",
        timestamp="2026-10-20T12:00:00Z",
        source_id="camera_01",
        risk_level="CRITICAL",
        risk_score=0.9,
        total_area_ratio=0.2,
        growth_derivative=0.1,
        spread_direction="Static",
        hazard_count=1,
        message="Retry exhausted test",
    )

    with patch("connector.alert.requests.post", side_effect=requests.RequestException("Connection failed")) as mock_post, \
         patch("connector.alert.time.sleep") as mock_sleep:
        success = connector._do_send(payload)

        assert success is False
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)


def test_alert_connector_disabled_does_not_dispatch():
    """When enabled=False, connector must never dispatch alerts."""
    connector = AlertConnector(enabled=False, simulate_only=True)
    assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.99,
        total_area_ratio=0.5,
        smoke_area_ratio=0.2,
        fire_area_ratio=0.3,
        growth_derivative=0.2,
        spread_direction="Static",
        active_hazard_count=2,
        message="Disabled connector test",
    )
    assert connector.maybe_dispatch(assessment) is False


def test_alert_connector_cooldown_expires_and_rearms():
    """After cooldown period has elapsed, the same risk level should be dispatched again."""
    current_time = 1000.0

    def mock_clock():
        return current_time

    connector = AlertConnector(enabled=True, cooldown_seconds=10.0, simulate_only=True, time_provider=mock_clock)
    assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.9,
        total_area_ratio=0.2,
        smoke_area_ratio=0.1,
        fire_area_ratio=0.1,
        growth_derivative=0.1,
        spread_direction="Static",
        active_hazard_count=1,
        message="Cooldown expiry test",
    )
    assert connector.maybe_dispatch(assessment) is True
    # Immediate is suppressed
    assert connector.maybe_dispatch(assessment) is False
    # Advance time past cooldown
    current_time += 11.0
    assert connector.maybe_dispatch(assessment) is True


def test_alert_connector_warning_to_critical_escalation_bypasses_warning_cooldown():
    """An immediate escalation from WARNING to CRITICAL must not be blocked by WARNING cooldown."""
    connector = AlertConnector(enabled=True, cooldown_seconds=10.0, simulate_only=True)
    warn_assessment = RiskAssessment(
        level=RiskLevel.WARNING,
        score=0.6,
        total_area_ratio=0.05,
        smoke_area_ratio=0.05,
        fire_area_ratio=0.0,
        growth_derivative=0.08,
        spread_direction="Static",
        active_hazard_count=1,
        message="Warning assessment",
    )
    crit_assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.95,
        total_area_ratio=0.2,
        smoke_area_ratio=0.1,
        fire_area_ratio=0.1,
        growth_derivative=0.2,
        spread_direction="Upward",
        active_hazard_count=2,
        message="Critical escalation",
    )
    # WARNING dispatched
    assert connector.maybe_dispatch(warn_assessment) is True
    # Immediate escalation to CRITICAL should dispatch without being blocked by WARNING cooldown
    assert connector.maybe_dispatch(crit_assessment) is True


def test_alert_connector_simulate_only_makes_no_http_call():
    """When simulate_only=True, no network/HTTP requests should be executed."""
    connector = AlertConnector(
        enabled=True,
        simulate_only=True,
        webhook_url="http://localhost:8080/api/v1/alerts",
    )
    payload = AlertPayload(
        event_id="ALERT-SIM-01",
        timestamp="2026-10-20T12:00:00Z",
        source_id="camera_01",
        risk_level="CRITICAL",
        risk_score=0.9,
        total_area_ratio=0.2,
        growth_derivative=0.1,
        spread_direction="Static",
        hazard_count=1,
        message="Simulation test",
    )
    with patch("connector.alert.requests.post") as mock_post:
        success = connector._do_send(payload)
        assert success is True
        mock_post.assert_not_called()


def test_alert_connector_client_error_4xx_does_not_retry():
    """HTTP 4xx (e.g. 400 Bad Request, 401 Unauthorized) should fail immediately without retry loops."""
    connector = AlertConnector(
        enabled=True,
        simulate_only=False,
        webhook_url="http://localhost:8080/api/v1/alerts",
        max_retries=3,
    )
    payload = AlertPayload(
        event_id="ALERT-4XX-01",
        timestamp="2026-10-20T12:00:00Z",
        source_id="camera_01",
        risk_level="CRITICAL",
        risk_score=0.9,
        total_area_ratio=0.2,
        growth_derivative=0.1,
        spread_direction="Static",
        hazard_count=1,
        message="4xx non-retry test",
    )
    mock_400 = MagicMock()
    mock_400.status_code = 400
    mock_400.text = "Bad Request"

    with patch("connector.alert.requests.post", return_value=mock_400) as mock_post, \
         patch("connector.alert.time.sleep") as mock_sleep:
        success = connector._do_send(payload)
        assert success is False
        # Should NOT retry on 400; call count must be exactly 1
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


def test_alert_connector_resolved_event_on_hazard_clearance():
    """When risk returns to NORMAL after an active alert, a RESOLVED event should be dispatched."""
    connector = AlertConnector(enabled=True, simulate_only=True)
    crit_assessment = RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=0.9,
        total_area_ratio=0.2,
        smoke_area_ratio=0.1,
        fire_area_ratio=0.1,
        growth_derivative=0.1,
        spread_direction="Static",
        active_hazard_count=1,
        message="Critical threat",
    )
    normal_assessment = RiskAssessment(
        level=RiskLevel.NORMAL,
        score=0.0,
        total_area_ratio=0.0,
        smoke_area_ratio=0.0,
        fire_area_ratio=0.0,
        growth_derivative=0.0,
        spread_direction="Static",
        active_hazard_count=0,
        message="All clear",
    )
    assert connector.maybe_dispatch(crit_assessment) is True
    # Clearing the hazard should dispatch a RESOLVED notification
    assert connector.maybe_dispatch(normal_assessment) is True


def test_alert_connector_stop_idempotent():
    """Calling stop() multiple times should not raise any exceptions."""
    connector = AlertConnector(enabled=True, simulate_only=True)
    connector.start()
    connector.stop()
    # Second stop call
    connector.stop()
    assert connector._worker is None


def test_alert_connector_stop_full_queue():
    """Stopping connector when queue is completely full should not crash."""
    connector = AlertConnector(enabled=True, simulate_only=True, max_queue_size=2)
    # Fill queue
    connector._queue.put("item1")
    connector._queue.put("item2")
    assert connector._queue.full()

    # Calling stop with full queue should execute cleanly without raising queue.Full
    connector.stop(timeout=0.2)


