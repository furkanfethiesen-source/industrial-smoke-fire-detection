"""Industrial Notification & Webhook Alert Connector.

Triggers REST webhooks or MQTT notifications when risk levels escalate to
WARNING or CRITICAL. Includes rate-limiting/cooldown to prevent alarm flooding.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable

import requests

from src.risk_analyzer import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    """Standardized event payload dispatched to security/MES/SCADA systems."""

    event_id: str
    timestamp: str
    source_id: str
    risk_level: str
    risk_score: float
    total_area_ratio: float
    growth_derivative: float
    spread_direction: str
    hazard_count: int
    message: str

    @classmethod
    def from_assessment(
        cls,
        assessment: RiskAssessment,
        source_id: str = "camera_01",
        event_type: Optional[str] = None,
    ) -> AlertPayload:
        now_iso = datetime.now(timezone.utc).isoformat()
        event_id = f"ALERT-{uuid.uuid4()}"
        risk_lvl = event_type if event_type else assessment.level.value
        return cls(
            event_id=event_id,
            timestamp=now_iso,
            source_id=source_id,
            risk_level=risk_lvl,
            risk_score=assessment.score,
            total_area_ratio=assessment.total_area_ratio,
            growth_derivative=assessment.growth_derivative,
            spread_direction=assessment.spread_direction,
            hazard_count=assessment.active_hazard_count,
            message=assessment.message,
        )



class AlertConnector:
    """Manages alert dispatching to webhooks or monitoring backends."""

    def __init__(
        self,
        enabled: bool = True,
        webhook_url: Optional[str] = "http://localhost:8080/api/v1/alerts",
        request_timeout_sec: float = 3.0,
        cooldown_seconds: float = 10.0,
        simulate_only: bool = True,
        source_id: str = "camera_01",
        max_queue_size: int = 100,
        max_retries: int = 3,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.enabled = enabled
        self.webhook_url = webhook_url
        self.request_timeout_sec = request_timeout_sec
        self.cooldown_seconds = cooldown_seconds
        self.simulate_only = simulate_only
        self.source_id = source_id
        self._time_provider: Callable[[], float] = time_provider or time.time

        self.last_dispatched_time: Dict[RiskLevel, float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._max_retries: int = max_retries


    def start(self) -> None:
        """Creates and starts daemon worker thread running _worker_loop."""
        if self._worker is not None and self._worker.is_alive():
            logger.warning("Worker thread is already running.")
            return

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AlertWorkerThread",
            daemon=True,
        )
        self._worker.start()
        logger.info("AlertConnector background worker thread started.")

    def stop(self, timeout: float = 10.0, drain: bool = True) -> None:
        """Sets stop event, optionally drains pending items, puts sentinel None, and joins thread with timeout."""
        if drain and self._worker is not None and self._worker.is_alive():
            # Allow worker to drain pending alerts before signaling stop
            drain_deadline = time.time() + max(1.0, timeout - 2.0)
            while not self._queue.empty() and time.time() < drain_deadline:
                time.sleep(0.05)

        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Worker loop checks self._stop_event on get timeout (0.5s), so it will cleanly exit even if full
            pass

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)
            if self._worker.is_alive():
                remaining = self._queue.qsize()
                logger.warning(
                    "Worker thread did not exit within timeout of %.1fs; %d alert(s) remain in queue.",
                    timeout,
                    remaining,
                )
            else:
                logger.info("AlertConnector background worker thread stopped.")
            self._worker = None

    def _worker_loop(self) -> None:
        """Worker thread loop consuming payloads from queue and dispatching them."""
        while True:
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if payload is None:
                self._queue.task_done()
                break

            try:
                self._do_send(payload)
            except Exception as exc:
                logger.error("Unexpected error in alert worker loop: %s", exc)
            finally:
                self._queue.task_done()

    def _do_send(self, payload: AlertPayload) -> bool:
        """Sends payload to endpoint or simulates notification with retry logic."""
        data_dict = asdict(payload)

        if self.simulate_only or not self.webhook_url:
            logger.warning(
                "🚨 [SIMULATED ALARM DISPATCH] -> Level: %s | Msg: %s | Payload: %s",
                payload.risk_level,
                payload.message,
                json.dumps(data_dict),
            )
            return True

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=data_dict,
                    timeout=self.request_timeout_sec,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code in (200, 201, 202):
                    logger.info("Successfully dispatched %s alarm to %s", payload.risk_level, self.webhook_url)
                    return True
                elif 400 <= resp.status_code < 500:
                    # Client-side errors (400, 401, 403, 404): Retrying is futile and floods the endpoint
                    logger.error(
                        "Client error dispatching alert to %s (Status: %d, Response: %s). Non-retryable.",
                        self.webhook_url,
                        resp.status_code,
                        resp.text,
                    )
                    return False
                else:
                    logger.error(
                        "Server error dispatching alert to %s (Status: %d, Response: %s). Retrying...",
                        self.webhook_url,
                        resp.status_code,
                        resp.text,
                    )
            except Exception as exc:
                logger.error("Error dispatching alert to %s: %s", self.webhook_url, exc)

            if attempt < self._max_retries - 1:
                backoff = 0.5 * (2 ** attempt)
                time.sleep(backoff)

        return False

    def _enqueue(self, payload: AlertPayload) -> bool:
        """Puts payload into queue (non-blocking) with drop-oldest policy if full."""
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                dropped_id = getattr(dropped, "event_id", "unknown") if dropped is not None else "sentinel"
                dropped_level = getattr(dropped, "risk_level", "unknown") if dropped is not None else "none"
                logger.warning(
                    "Alert queue full. Dropped oldest alert (event_id=%s, level=%s) to enqueue new alert.",
                    dropped_id,
                    dropped_level,
                )
            except queue.Empty:
                pass

            try:
                self._queue.put_nowait(payload)
                return True
            except queue.Full:
                logger.warning("Failed to enqueue alert even after drop-oldest.")
                return False

    def _send(self, payload: AlertPayload) -> bool:
        """Backwards compatibility alias for _enqueue."""
        return self._enqueue(payload)

    def maybe_dispatch(self, assessment: RiskAssessment) -> bool:
        """Dispatches alert if current risk level meets threshold and cooldown elapsed."""
        if not self.enabled:
            return False

        # Check if an active alarm has cleared -> dispatch RESOLVED lifecycle event
        has_active_alarm = any(
            self.last_dispatched_time.get(lvl, 0.0) > 0.0
            for lvl in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        )

        if assessment.level not in (RiskLevel.WARNING, RiskLevel.CRITICAL):
            if has_active_alarm:
                # Clear all active alarm timestamps and dispatch RESOLVED
                self.last_dispatched_time.clear()
                payload = AlertPayload.from_assessment(
                    assessment,
                    source_id=self.source_id,
                    event_type="RESOLVED",
                )
                payload.message = f"Hazard resolved. System returned to {assessment.level.value}."
                return self._enqueue(payload)
            return False

        now = self._time_provider()
        last_time = self.last_dispatched_time.get(assessment.level, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            logger.debug(
                "Suppressed %s alert due to cooldown (%.1fs remaining)",
                assessment.level.value,
                self.cooldown_seconds - (now - last_time),
            )
            return False

        payload = AlertPayload.from_assessment(assessment, source_id=self.source_id)
        dispatched = self._enqueue(payload)
        if dispatched:
            self.last_dispatched_time[assessment.level] = now

        return dispatched

