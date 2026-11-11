"""Unit tests for ThreadedStream zero-lag video capture and reconnection."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.threaded_stream import ThreadedStream


@pytest.fixture
def dummy_video_path(tmp_path: Path) -> Path:
    """Creates a temporary dummy video file using cv2.VideoWriter."""
    video_file = tmp_path / "dummy_sample.mp4"
    width, height, fps = 64, 64, 30.0
    num_frames = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to initialize VideoWriter for {video_file}"

    for i in range(num_frames):
        # Create a frame with unique pixel values
        color_val = int((i * 10) % 256)
        frame = np.full((height, width, 3), fill_value=color_val, dtype=np.uint8)
        # Add a colored rectangle in the center
        cv2.rectangle(frame, (16, 16), (48, 48), (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    assert video_file.exists(), f"Video file was not created: {video_file}"
    return video_file


def test_read_from_real_video(dummy_video_path: Path):
    """1. Test that reading from a real video file works."""
    stream = ThreadedStream(str(dummy_video_path))
    assert stream.fps == pytest.approx(30.0, rel=1e-1)

    # Before start, buffer is empty
    ret, frame = stream.read()
    assert ret is False
    assert frame is None

    # Start background capture
    ret_self = stream.start()
    assert ret_self is stream
    assert stream.is_alive() is True

    # Poll until a frame is available
    ret = False
    captured_frame = None
    start_time = time.time()
    while time.time() - start_time < 2.0:
        ret, captured_frame = stream.read()
        if ret and captured_frame is not None:
            break
        time.sleep(0.01)

    assert ret is True
    assert captured_frame is not None
    assert isinstance(captured_frame, np.ndarray)
    assert captured_frame.shape == (64, 64, 3)
    assert captured_frame.dtype == np.uint8

    # Clean up
    stream.stop()
    assert stream.is_alive() is False


def test_stop_properly_terminates_thread(dummy_video_path: Path):
    """2. Test that stop() properly terminates the background thread."""
    stream = ThreadedStream(str(dummy_video_path))
    stream.start()
    assert stream.is_alive() is True

    # Stop should join thread and release capture
    stream.stop()
    assert stream.is_alive() is False
    assert stream._thread is not None
    assert stream._thread.is_alive() is False

    # Calling stop() again should be safely idempotent
    stream.stop()
    assert stream.is_alive() is False

    # Reading after stop should return (False, None)
    ret, frame = stream.read()
    assert ret is False
    assert frame is None


def test_is_alive_state_transitions(dummy_video_path: Path):
    """3. Test that is_alive() returns correct state before/after start/stop."""
    stream = ThreadedStream(str(dummy_video_path))

    # Before start
    assert stream.is_alive() is False

    # After start
    stream.start()
    assert stream.is_alive() is True

    # After stop
    stream.stop()
    assert stream.is_alive() is False


def test_file_source_not_found_fails_fast(tmp_path: Path):
    """File sources that do not exist must fail immediately with FileNotFoundError without reconnecting."""
    invalid_file = tmp_path / "nonexistent_source_stream_000.mp4"
    assert not invalid_file.exists()

    with pytest.raises(FileNotFoundError, match="Video file not found"):
        ThreadedStream(source=str(invalid_file))


def test_invalid_live_source_triggers_reconnect(caplog: pytest.LogCaptureFixture):
    """Invalid live RTSP stream triggers reconnect attempts (state & logging)."""
    invalid_rtsp = "rtsp://invalid_host_address:8554/live"

    with caplog.at_level(logging.WARNING):
        stream = ThreadedStream(
            source=invalid_rtsp,
            reconnect_max=5,
            backoff_base=0.01,
            backoff_max=0.05,
        )

        assert stream.reconnect_attempts == 0
        assert stream.reconnect_count == 0
        assert stream.fps == 0.0

        stream.start()
        assert stream.is_alive() is True

        # Wait briefly for worker thread to attempt reconnection
        start_time = time.time()
        while time.time() - start_time < 1.0:
            if stream.reconnect_attempts >= 1:
                break
            time.sleep(0.01)

        # Verify state
        assert stream.reconnect_attempts >= 1
        assert stream.reconnect_count >= 1

        # Verify logging
        reconnect_logged = any(
            "reconnect" in record.message.lower() for record in caplog.records
        )
        assert reconnect_logged, f"Expected reconnect log in: {[r.message for r in caplog.records]}"

        # Graceful stop
        stream.stop()
        assert stream.is_alive() is False


def test_reconnect_max_exhaustion_terminates_thread():
    """Test that exceeding reconnect_max terminates the background thread and sets has_failed."""
    invalid_rtsp = "rtsp://invalid_host_address_exhaust:8554/live"

    stream = ThreadedStream(
        source=invalid_rtsp,
        reconnect_max=2,
        backoff_base=0.01,
        backoff_max=0.02,
    )
    stream.start()

    # Wait for the thread to exceed max attempts and terminate
    start_time = time.time()
    while time.time() - start_time < 2.0:
        if not stream.is_alive():
            break
        time.sleep(0.02)

    # Thread should have self-terminated after exceeding reconnect_max
    assert stream.is_alive() is False
    assert stream.reconnect_attempts > stream.reconnect_max
    assert stream.reconnect_exhausted is True
    assert stream.has_failed is True
    stream.stop()


def test_context_manager(dummy_video_path: Path):
    """Test using ThreadedStream as a context manager."""
    with ThreadedStream(str(dummy_video_path)) as stream:
        assert stream.is_alive() is True
        start_time = time.time()
        ret = False
        while time.time() - start_time < 2.0:
            ret, _ = stream.read()
            if ret:
                break
            time.sleep(0.01)
        assert ret is True

    # Exiting context manager should stop the stream
    assert stream.is_alive() is False


def test_file_source_eof_stops_cleanly_without_infinite_loop(dummy_video_path: Path):
    """When reading from a video file source, reaching EOF must not loop and reopen the file."""
    stream = ThreadedStream(str(dummy_video_path), reconnect_max=5)
    assert stream.is_file_source is True
    stream.start()

    # Wait until all frames are played and EOF is reached
    start_time = time.time()
    while time.time() - start_time < 3.0:
        if stream.is_eof:
            break
        time.sleep(0.05)

    assert stream.is_eof is True
    # Crucially, stream should NOT reconnect or loop back to frame 0
    assert stream.reconnect_attempts == 0
    # Reading after EOF must return ret=False or signal EOF
    ret, frame = stream.read()
    # Stopping should be clean
    stream.stop()
    assert stream.is_alive() is False


def test_frame_consumption_flag():
    """Consecutive read_with_status() calls when no new frame arrived should indicate frame is already consumed."""
    stream = ThreadedStream("rtsp://dummy_live_source:8554/live")
    dummy_frame = np.zeros((32, 32, 3), dtype=np.uint8)

    # 1. Henüz kare yokken okuma
    ret, frame, is_new = stream.read_with_status()
    assert ret is False
    assert is_new is False

    # 2. Yeni kare arabelleğe eklendiğinde
    with stream._lock:
        stream._frame_buffer.append(dummy_frame)
        stream._latest_frame_consumed = False
        stream.frame_number += 1

    ret, frame, is_new = stream.read_with_status()
    assert ret is True
    assert is_new is True

    # 3. Aynı kare henüz yenilenmeden tekrar okunduğunda is_new=False olmalı
    ret_second, _, is_new_second = stream.read_with_status()
    assert ret_second is True
    assert is_new_second is False, "Same frame should not be flagged as is_new"

    # 4. Yeni bir kare geldiğinde tekrar is_new=True olmalı
    with stream._lock:
        stream._frame_buffer.append(dummy_frame)
        stream._latest_frame_consumed = False
        stream.frame_number += 1

    ret_third, _, is_new_third = stream.read_with_status()
    assert ret_third is True
    assert is_new_third is True


def test_file_source_sequential_reading_no_frame_drop(dummy_video_path: Path):
    """Ensure all frames from video file are read sequentially without dropping any frame."""
    stream = ThreadedStream(str(dummy_video_path)).start()

    read_count = 0
    while True:
        ret, frame, is_new = stream.read_with_status()
        if not ret or frame is None:
            if stream.is_eof and stream.is_empty():
                break
            time.sleep(0.01)
            continue
        assert is_new is True
        read_count += 1

    stream.stop()
    # dummy_video_path fixture generates 25 frames
    assert read_count == 25, f"Expected 25 frames sequentially read, got {read_count}"


def test_last_read_frame_idx_sequential(dummy_video_path: Path):
    """Verify stream.last_read_frame_idx advances strictly 1..N matching each consumed frame from queue."""
    stream = ThreadedStream(str(dummy_video_path)).start()

    consumed_indices = []
    while True:
        ret, frame, is_new = stream.read_with_status()
        if not ret or frame is None:
            if stream.is_eof and stream.is_empty():
                break
            time.sleep(0.01)
            continue
        consumed_indices.append(stream.last_read_frame_idx)

    stream.stop()
    assert consumed_indices == list(range(1, 26)), f"Consumed indices mismatched: {consumed_indices}"




