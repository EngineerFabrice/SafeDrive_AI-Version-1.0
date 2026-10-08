"""Monitoring state resolution, bounding-box geometry and driver ROI selection."""
import pytest

from engine.detectors import BoundingBox
from engine.detectors.driver_roi import DriverROIConfig, DriverROISelector
from engine.detectors.person import PersonDetection
from engine.state import CameraStatus, ModelStatus, MonitoringState, MonitoringStatus, resolve_status

C, R = CameraStatus.CONNECTED, ModelStatus.READY


@pytest.mark.parametrize("args, expected", [
    ((False, C, R, True, True, True), MonitoringStatus.STOPPED),
    ((True, CameraStatus.UNAVAILABLE, R, True, True, True), MonitoringStatus.CAMERA_UNAVAILABLE),
    ((True, C, ModelStatus.UNAVAILABLE, True, True, True), MonitoringStatus.MODEL_UNAVAILABLE),
    ((True, C, R, False, True, True), MonitoringStatus.NO_DRIVER),
    ((True, C, R, True, False, True), MonitoringStatus.FACE_NOT_DETECTED),
    ((True, C, R, True, True, False), MonitoringStatus.FEATURES_UNAVAILABLE),
    ((True, C, R, True, True, True), MonitoringStatus.MONITORING_READY),
])
def test_resolve_status(args, expected):
    assert resolve_status(*args) == expected


def test_most_fundamental_failure_wins():
    # camera down AND no driver: the camera problem is reported, never masked
    assert resolve_status(True, CameraStatus.UNAVAILABLE, ModelStatus.UNAVAILABLE,
                          False, False, False) == MonitoringStatus.CAMERA_UNAVAILABLE


def test_state_snapshot_is_immutable_and_serialisable():
    state = MonitoringState()
    snap = state.update(status=MonitoringStatus.NO_DRIVER, fps=9.5)
    data = snap.to_dict()
    assert data["status"] == "NO_DRIVER" and data["fps"] == 9.5
    with pytest.raises(Exception):
        snap.fps = 1.0


def test_bounding_box_geometry():
    a, b = BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)
    assert a.area == 100 and a.center == (5, 5)
    assert a.iou(b) == pytest.approx(25 / 175)
    assert BoundingBox(-5, -5, 50, 50).clip(20, 20).as_tuple() == (0, 0, 20, 20)
    assert BoundingBox(10, 10, 20, 20).expand(0.5).as_tuple() == (5, 5, 25, 25)


def _person(x1, y1, x2, y2, conf=0.9):
    return PersonDetection(BoundingBox(x1, y1, x2, y2), conf)


def test_roi_prefers_large_central_person():
    sel = DriverROISelector()
    driver = _person(200, 80, 440, 480)          # large, central
    passenger = _person(520, 300, 600, 420)      # small, at the edge
    roi = sel.select([passenger, driver], (480, 640, 3))
    assert roi.person is driver
    assert roi.roi.area >= driver.bbox.area      # padded so the face is not cut off


def test_roi_rejects_tiny_people_and_handles_empty():
    sel = DriverROISelector(DriverROIConfig(min_area_fraction=0.04))
    assert sel.select([_person(0, 0, 20, 20)], (480, 640, 3)) is None
    assert sel.select([], (480, 640, 3)) is None
