"""Zero-lag threaded video stream capture engine.

Runs cv2.VideoCapture in a dedicated background daemon thread to continuously
drain frames, keeping latency at zero by holding only the latest frame in a
single-element deque. Implements exponential backoff reconnection when frames
fail to read or stream disconnects.
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedStream:
    """Threaded video stream reader with zero-lag buffer and auto-reconnect.

    For live streams (RTSP, webcam), it drains frames in a background daemon thread
    into a `collections.deque(maxlen=1)` buffer so callers always retrieve the freshest
    available frame without accumulating latency.

    For video file sources, it uses a bounded queue (`queue.Queue`) without dropping
    any frames, ensuring sequential, deterministic processing for optical flow and analytics.

    Attributes:
        source: Source device index or stream URI / file path.
        reconnect_max: Maximum consecutive reconnection attempts before giving up.
        backoff_base: Initial backoff delay in seconds (default: 1.0s).
        backoff_max: Maximum backoff delay cap in seconds (default: 30.0s).
        reconnect_attempts: Current count of consecutive failed reads/reconnections.
    """

    def __init__(
        self,
        source: Union[int, str],
        reconnect_max: int = 50,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ) -> None:
        """Initializes the ThreadedStream instance.

        Args:
            source: Video source, e.g., camera index (0), RTSP URL, or file path.
            reconnect_max: Maximum number of reconnection retries before stopping.
            backoff_base: Initial reconnect delay in seconds (doubled each failure).
            backoff_max: Maximum backoff delay cap in seconds.
        """
        self.source = source
        self.reconnect_max = reconnect_max
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        # Source classification: file vs live stream/camera
        self.is_file_source: bool = False
        if isinstance(self.source, str):
            lower_src = self.source.lower()
            if not lower_src.startswith(("rtsp://", "http://", "https://", "rtmp://", "/dev/video")) and not self.source.isdigit():
                self.is_file_source = True

        # Fail-fast check for file sources: if local path is specified and missing, raise immediately
        if self.is_file_source and isinstance(self.source, str):
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")

        # Resolve source for cv2.VideoCapture (e.g. numeric string "0" -> int 0)
        self._source_for_cap: Union[int, str]
        if isinstance(source, int):
            self._source_for_cap = source
        elif isinstance(source, str) and source.isdigit():
            self._source_for_cap = int(source)
        else:
            self._source_for_cap = str(source)

        # Thread synchronization and frame storage
        # Live streams use deque(maxlen=1) for zero lag; file sources use queue.Queue for zero frame drop
        if self.is_file_source:
            self._frame_queue: Optional[queue.Queue] = queue.Queue(maxsize=128)
            self._frame_buffer = None
        else:
            self._frame_queue = None
            self._frame_buffer = collections.deque(maxlen=1)

        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Reconnection telemetry & status
        self.reconnect_attempts: int = 0
        self.reconnect_exhausted: bool = False
        self.has_failed: bool = False

        # Video capture handle and metadata
        self._fps: float = 0.0
        self._cap: Optional[cv2.VideoCapture] = None

        self.is_eof: bool = False
        self.frame_number: int = 0
        self.last_read_frame_idx: int = 0
        self._latest_frame_consumed: bool = False
        self.on_disconnect: Optional[callable] = None

        try:
            self._cap = cv2.VideoCapture(self._source_for_cap)
            if self._cap.isOpened():
                val = self._cap.get(cv2.CAP_PROP_FPS)
                if val and val > 0 and not np.isnan(val):
                    self._fps = float(val)
            elif self.is_file_source:
                # If VideoCapture failed to open a valid file path, fail fast
                raise FileNotFoundError(f"Video file cannot be opened by OpenCV: {self.source}")
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.warning("Failed to open capture in __init__ for '%s': %s", self.source, exc)

    @property
    def fps(self) -> float:
        """Source frame rate (FPS) reported by the video capture device."""
        with self._cap_lock:
            if self._fps <= 0.0 and self._cap is not None and self._cap.isOpened():
                val = self._cap.get(cv2.CAP_PROP_FPS)
                if val and val > 0 and not np.isnan(val):
                    self._fps = float(val)
        return self._fps

    @property
    def reconnect_count(self) -> int:
        """Alias for consecutive reconnect attempts."""
        return self.reconnect_attempts

    def is_empty(self) -> bool:
        """Checks if internal frame buffer/queue is empty."""
        if self.is_file_source and self._frame_queue is not None:
            return self._frame_queue.empty()
        with self._lock:
            return len(self._frame_buffer) == 0 if self._frame_buffer is not None else True

    def start(self) -> ThreadedStream:
        """Starts the background daemon thread to continuously read frames.

        Returns:
            ThreadedStream: Self instance for method chaining.
        """
        with self._cap_lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning("ThreadedStream background thread is already running.")
                return self

            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._source_for_cap)
                if self._cap.isOpened():
                    val = self._cap.get(cv2.CAP_PROP_FPS)
                    if val and val > 0 and not np.isnan(val):
                        self._fps = float(val)
                elif self.is_file_source:
                    raise FileNotFoundError(f"Cannot open video file: {self.source}")

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"ThreadedStream-{self.source}",
                daemon=True,
            )
            self._thread.start()
            logger.info("ThreadedStream background thread started for source '%s'.", self.source)
        return self

    def _capture_loop(self) -> None:
        """Continuous frame drain loop running inside the daemon thread."""
        logger.debug("Entering capture loop for source '%s'.", self.source)

        while not self._stop_event.is_set():
            ret = False
            frame = None

            cap_is_open = False
            with self._cap_lock:
                if self._cap is not None and self._cap.isOpened():
                    cap_is_open = True
                    try:
                        ret, frame = self._cap.read()
                    except Exception as exc:
                        logger.warning(
                            "Exception during cv2.VideoCapture.read() for source '%s': %s",
                            self.source,
                            exc,
                        )
                        ret = False
                        frame = None

            if ret and frame is not None:
                if self.reconnect_attempts > 0:
                    logger.info(
                        "Successfully recovered connection to source '%s' after %d attempt(s).",
                        self.source,
                        self.reconnect_attempts,
                    )
                    self.reconnect_attempts = 0

                with self._lock:
                    self.frame_number += 1
                    self._latest_frame_consumed = False
                    current_idx = self.frame_number

                if self.is_file_source and self._frame_queue is not None:
                    # Sequential blocking queue: store (frame_idx, frame) tuple
                    while not self._stop_event.is_set():
                        try:
                            self._frame_queue.put((current_idx, frame), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                else:
                    with self._lock:
                        if self._frame_buffer is not None:
                            self._frame_buffer.append(frame)
            else:
                # If reading from a static file source that was open and we read at least 1 frame (or opened successfully),
                # failing to read next frame signifies EOF.
                if self.is_file_source:
                    logger.info("Reached end-of-file (EOF) for source '%s'. Terminating capture loop.", self.source)
                    self.is_eof = True
                    break

                # For live streams/cameras: notify disconnect callback if configured
                if self.reconnect_attempts == 0 and self.on_disconnect is not None:
                    try:
                        self.on_disconnect(self.source)
                    except Exception as exc:
                        logger.warning("Error in on_disconnect callback: %s", exc)

                # Read failed or capture not open -> trigger exponential backoff reconnect
                self.reconnect_attempts += 1
                if self.reconnect_attempts > self.reconnect_max:
                    logger.critical(
                        "Exceeded maximum reconnect attempts (%d/%d) for source '%s'. Aborting capture loop.",
                        self.reconnect_attempts,
                        self.reconnect_max,
                        self.source,
                    )
                    self.reconnect_exhausted = True
                    self.has_failed = True
                    break

                # Exponential backoff: 1s -> 2s -> 4s -> ... capped at backoff_max
                delay = min(
                    self.backoff_base * (2.0 ** (self.reconnect_attempts - 1)),
                    self.backoff_max,
                )
                logger.warning(
                    "Frame read failed for source '%s'. Reconnect attempt %d/%d in %.2fs...",
                    self.source,
                    self.reconnect_attempts,
                    self.reconnect_max,
                    delay,
                )

                # Step 1: release current capture handle
                with self._cap_lock:
                    if self._cap is not None:
                        self._cap.release()

                # Step 2: wait backoff duration, allowing early termination on stop_event
                if self._stop_event.wait(timeout=delay):
                    logger.debug(
                        "Stop event signaled during reconnect backoff for '%s'. Exiting.",
                        self.source,
                    )
                    break

                # Step 3: reopen capture handle
                with self._cap_lock:
                    if self._cap is None:
                        self._cap = cv2.VideoCapture()
                    opened = self._cap.open(self._source_for_cap)
                    if opened and self._cap.isOpened():
                        val = self._cap.get(cv2.CAP_PROP_FPS)
                        if val and val > 0 and not np.isnan(val):
                            self._fps = float(val)
                        logger.info("Successfully reopened capture source '%s'.", self.source)
                    else:
                        logger.warning(
                            "Reconnect attempt %d failed to open source '%s'.",
                            self.reconnect_attempts,
                            self.source,
                        )

        # Cleanup on loop termination
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        logger.debug("Capture loop terminated for source '%s'.", self.source)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the next frame.

        For file sources: sequentially drains from queue, blocking briefly if necessary.
        For live sources: non-blocking retrieval of latest frame from single-element buffer.
        """
        if self.is_file_source and self._frame_queue is not None:
            while not self._stop_event.is_set():
                try:
                    f_idx, frame = self._frame_queue.get(timeout=0.05)
                    self.last_read_frame_idx = f_idx
                    self._latest_frame_consumed = True
                    return True, frame
                except queue.Empty:
                    if self.is_eof or not self.is_alive():
                        return False, None
            return False, None

        with self._lock:
            if self._frame_buffer:
                self.last_read_frame_idx = self.frame_number
                self._latest_frame_consumed = True
                frame = self._frame_buffer[0]
                return True, frame
            return False, None

    def read_with_status(self) -> Tuple[bool, Optional[np.ndarray], bool]:
        """Retrieves the next frame along with an is_new flag.

        For file sources: sequentially drains from queue. Each frame returned is new.
        For live sources: returns latest frame and indicates whether it was unconsumed.
        """
        if self.is_file_source and self._frame_queue is not None:
            while not self._stop_event.is_set():
                try:
                    f_idx, frame = self._frame_queue.get(timeout=0.05)
                    self.last_read_frame_idx = f_idx
                    self._latest_frame_consumed = True
                    return True, frame, True
                except queue.Empty:
                    if self.is_eof or not self.is_alive():
                        return False, None, False
            return False, None, False

        with self._lock:
            if self._frame_buffer:
                self.last_read_frame_idx = self.frame_number
                is_new = not self._latest_frame_consumed
                self._latest_frame_consumed = True
                frame = self._frame_buffer[0]
                return True, frame, is_new
            return False, None, False

    def stop(self) -> None:
        """Gracefully stops the background thread, joins it, and releases the capture.

        Behavior Notes:
            - Under normal EOF operation: Consumers should read until `is_eof` and `is_empty()`
              are True to process all available frames.
            - Under early exit / cancellation: When `stop()` is explicitly called before EOF,
              the worker thread is terminated immediately and any remaining unconsumed frames
              in `_frame_queue` or `_frame_buffer` are cleared/dropped. In test suites or batch
              processors where 100% of frames must be asserted, ensure `is_empty()` is True
              before invoking `stop()`.
        """
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=5.0)

        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    logger.debug("Exception releasing VideoCapture in stop(): %s", exc)
                self._cap = None

        with self._lock:
            if self._frame_buffer is not None:
                self._frame_buffer.clear()
            if self._frame_queue is not None:
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break

        logger.info("ThreadedStream stopped for source '%s'.", self.source)

    def is_alive(self) -> bool:
        """Checks if the background capture thread is alive and running.

        Returns:
            bool: True if the background thread is running, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> ThreadedStream:
        """Context manager entry point; starts the stream."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point; stops the stream."""
        self.stop()

    def __repr__(self) -> str:
        status = "alive" if self.is_alive() else "stopped"
        return f"<ThreadedStream source={self.source!r} status={status} fps={self.fps:.1f}>"
