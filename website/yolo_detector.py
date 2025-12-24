# website/yolo_detector.py
from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (person detection)
yolo_model = YOLO("yolov8n.pt")  # small YOLOv8 model

def detect_person(frame):
    """
    Detect persons in a frame and return cropped face/person images.
    Returns a list of cropped images and bounding boxes.
    """
    results = yolo_model.predict(frame, verbose=False)
    cropped_persons = []
    for r in results:
        for box in r.boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            cropped_persons.append((cropped, (x1, y1, x2, y2)))
    return cropped_persons
