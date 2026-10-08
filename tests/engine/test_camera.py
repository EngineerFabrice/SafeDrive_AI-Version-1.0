"""Camera capture thread: failure to open, disconnect, reconnect (fake cv2.VideoCapture)."""
import threading
import time

import numpy as np
import pytest

import engine.camera as camera_module
from engine.camera import Camera, CameraConfig, FpsMeter
from engine.state import CameraStatus


class FakeDevice:
    """Shared, scriptable behaviour for every FakeCapture the camera opens."""
    def __init__(self):
        self.lock = threading.Lock()
        self.can_open = True
        self.delivering = True
        self.opens = 0


class FakeCapture:
    device: FakeDevice = None

    def __init__(self, source, backend=None):
        with self.device.lock:
            self.device.opens += 1
            self._open = self.device.can_open

    def isOpened(self):
        return self._open

    def set(self, *args):
        return True

    def read(self):
        time.sleep(0.005)
        if self._open and self.device.delivering:
            return True, np.zeros((48, 64, 3), np.uint8)
        return False, None

    def release(self):
        self._open = False


def wait_until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def device(monkeypatch):
    dev = FakeDevice()
    FakeCapture.device = dev
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", FakeCapture)
    return dev


def make_camera():
    return Camera(CameraConfig(read_failure_threshold=3, reconnect_delay=0.01, max_reconnect_delay=0.05,
                               stale_frame_timeout=0.3))


def test_camera_delivers_fresh_frames(device):
    cam = make_camera()
    cam.start()
    try:
        assert wait_until(lambda: cam.status == CameraStatus.CONNECTED)
        f1 = cam.wait_for_frame(timeout=1.0)
        f2 = cam.wait_for_frame(after_id=f1.frame_id, timeout=1.0)
        assert f2.frame_id > f1.frame_id and f2.image.shape == (48, 64, 3)
    finally:
        cam.stop()
    assert cam.status == CameraStatus.STOPPED and not cam.is_running


def test_camera_unavailable_when_device_cannot_open_then_recovers(device):
    device.can_open = False
    cam = make_camera()
    cam.start()
    try:
        assert wait_until(lambda: cam.status == CameraStatus.UNAVAILABLE)
        assert "Cannot open camera" in cam.last_error
        device.can_open = True
        assert wait_until(lambda: cam.status == CameraStatus.CONNECTED)
    finally:
        cam.stop()


def test_camera_disconnect_and_reconnect(device):
    cam = make_camera()
    cam.start()
    try:
        assert wait_until(lambda: cam.status == CameraStatus.CONNECTED)
        last = cam.latest_frame().frame_id
        opens_before = device.opens
        device.delivering = False                      # unplugged: reads start failing
        assert wait_until(lambda: cam.status == CameraStatus.UNAVAILABLE)
        device.delivering = True                       # plugged back in
        assert wait_until(lambda: cam.status == CameraStatus.CONNECTED and cam.latest_frame().frame_id > last)
        assert device.opens > opens_before             # the device was actually reopened
    finally:
        cam.stop()


def test_fps_meter():
    meter = FpsMeter(window=2.0)
    base = time.perf_counter()
    for i in range(11):                                # 11 ticks, 0.1 s apart, ending now
        meter.tick(now=base - 1.0 + i * 0.1)
    assert meter.fps == pytest.approx(10.0, rel=0.01)
    meter.reset()
    assert meter.fps == 0.0
