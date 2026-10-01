# website/yolo_detector.py
"""DEPRECATED: superseded by engine.detectors.person / engine.detectors.driver_roi.

Kept for reference only; nothing in the application imports it any more.
It returned boxes for *every* YOLO class (not just persons) and loaded the
model at import time. Use the engine's PersonDetector + DriverROISelector.
"""
import warnings

warnings.warn("website.yolo_detector is deprecated; use engine.detectors.person",
              DeprecationWarning, stacklevel=2)

_yolo_model = None


def detect_person(frame):
    """
    Detect objects in a frame and return cropped images with bounding boxes.
    Returns a list of (cropped image, (x1, y1, x2, y2)).
    """
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    results = _yolo_model.predict(frame, verbose=False)
    cropped_persons = []
    for r in results:
        for box in r.boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            cropped_persons.append((cropped, (x1, y1, x2, y2)))
    return cropped_persons
