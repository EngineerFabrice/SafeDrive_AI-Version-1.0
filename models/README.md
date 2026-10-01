# Models

## face_detection_yunet_2023mar.onnx

YuNet face detector with 5 facial landmarks, used by `engine/detectors/landmarks.py`
(primary face detector inside the driver ROI, and the landmark source for
`engine/features`).

- Source: OpenCV Zoo, `models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- License: MIT, Copyright (c) 2020 Shiqi Yu (upstream `models/face_detection_yunet/LICENSE`)
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Size: 232,589 bytes; loaded with `cv2.FaceDetectorYN` (OpenCV >= 4.8).

If the file is missing, face detection falls back to OpenCV's Haar cascade and
features are reported as `FEATURES_UNAVAILABLE` (no landmarks).

## yolov8n.pt (not committed)

Person detector weights (Ultralytics YOLOv8n, AGPL-3.0). Not stored in the
repository (`*.pt` is gitignored); Ultralytics downloads it on first use into
the working directory. Set `SAFEDRIVE_YOLO_WEIGHTS` to use another path.
