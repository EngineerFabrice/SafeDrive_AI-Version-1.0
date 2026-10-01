"""Continuous OpenCV camera capture, independent of Flask.

A background thread reads frames as fast as the camera delivers them and
keeps only the most recent one, so consumers always process a fresh frame
instead of working through a backlog.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from .state import CameraStatus

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source: Union[int, str] = 0           # device index or video file / stream URL
    backend: int = cv2.CAP_ANY            # e.g. cv2.CAP_DSHOW / cv2.CAP_MSMF on Windows
    width: Optional[int] = None
    height: Optional[int] = None
    target_fps: Optional[float] = None
    read_failure_threshold: int = 10      # consecutive failed reads before treating as disconnected
    reconnect_delay: float = 1.0          # seconds; doubles after each failed attempt
    max_reconnect_delay: float = 10.0
    stale_frame_timeout: float = 2.0      # no new frame for this long => camera unavailable


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    frame_id: int
    captured_at: float   # time.perf_counter() when read() returned


class FpsMeter:
    """Rolling frames-per-second over a short time window."""

    def __init__(self, window: float = 2.0):
        self._window = window
        self._ticks = deque()
        self._lock = threading.Lock()

    def tick(self, now: Optional[float] = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._ticks.append(now)
            self._trim(now)

    def reset(self) -> None:
        with self._lock:
            self._ticks.clear()

    @property
    def fps(self) -> float:
        now = time.perf_counter()
        with self._lock:
            self._trim(now)
            if len(self._ticks) < 2:
                return 0.0
            span = self._ticks[-1] - self._ticks[0]
            return (len(self._ticks) - 1) / span if span > 0 else 0.0

    def _trim(self, now: float) -> None:
        while self._ticks and now - self._ticks[0] > self._window:
            self._ticks.popleft()


class Camera:
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_ready = threading.Condition()
        self._latest: Optional[Frame] = None
        self._next_id = 0
        self._status = CameraStatus.STOPPED
        self._last_error = ""
        self._fps = FpsMeter()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._latest = None
        self._fps.reset()
        self._status = CameraStatus.CONNECTING
        self._thread = threading.Thread(target=self._run, name="safedrive-camera", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("Camera thread did not exit within %.1fs", timeout)
            self._thread = None
        self._status = CameraStatus.STOPPED
        self._fps.reset()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ status
    @property
    def status(self) -> CameraStatus:
        # A device can stay "open" while delivering nothing (e.g. unplugged
        # mid-read on some backends), so stale frames also count as unavailable.
        if self._status == CameraStatus.CONNECTED and self._latest is not None:
            if time.perf_counter() - self._latest.captured_at > self.config.stale_frame_timeout:
                return CameraStatus.UNAVAILABLE
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def fps(self) -> float:
        return self._fps.fps

    # ------------------------------------------------------------ frames
    def latest_frame(self) -> Optional[Frame]:
        return self._latest

    def wait_for_frame(self, after_id: Optional[int] = None, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame newer than ``after_id`` exists; None on timeout or stop."""
        def fresh():
            return self._latest is not None and (after_id is None or self._latest.frame_id > after_id)

        with self._frame_ready:
            self._frame_ready.wait_for(lambda: fresh() or self._stop_event.is_set(), timeout)
            return self._latest if fresh() else None

    # ------------------------------------------------------------ internals
    def _open(self) -> bool:
        cfg = self.config
        cap = cv2.VideoCapture(cfg.source, cfg.backend)
        if not cap.isOpened():
            cap.release()
            return False
        if cfg.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.target_fps:
            cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prefer fresh frames; ignored by some backends
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run(self) -> None:
        delay = self.config.reconnect_delay
        failures = 0
        try:
            while not self._stop_event.is_set():
                if self._cap is None:
                    self._status = CameraStatus.CONNECTING
                    if not self._open():
                        self._status = CameraStatus.UNAVAILABLE
                        self._last_error = f"Cannot open camera source {self.config.source!r}"
                        log.warning("%s; retrying in %.1fs", self._last_error, delay)
                        self._stop_event.wait(delay)
                        delay = min(delay * 2, self.config.max_reconnect_delay)
                        continue
                    log.info("Camera %r opened", self.config.source)
                    delay = self.config.reconnect_delay
                    failures = 0

                ok, image = self._cap.read()
                if not ok or image is None:
                    failures += 1
                    if failures >= self.config.read_failure_threshold:
                        self._last_error = "Camera stopped delivering frames (disconnected?)"
                        log.warning(self._last_error)
                        self._status = CameraStatus.UNAVAILABLE
                        self._release()
                        self._stop_event.wait(delay)  # back off before reconnecting
                    else:
                        self._stop_event.wait(0.01)
                    continue

                failures = 0
                now = time.perf_counter()
                self._fps.tick(now)
                with self._frame_ready:
                    self._next_id += 1
                    self._latest = Frame(image=image, frame_id=self._next_id, captured_at=now)
                    self._status = CameraStatus.CONNECTED
                    self._last_error = ""
                    self._frame_ready.notify_all()
        finally:
            self._release()
